import os
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from sklearn.cluster import KMeans
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from vqvae import VQVAE


def read_config(file_path):
    """
    Read VQVAE model config file

    :param file_path str: config file path
    """
    with open(file_path, "r") as file:
        try:
            config = yaml.safe_load(file)
            model_config = config["model_config"]

            # Unpack the nested dictionaries
            architecture = SimpleNamespace(**model_config["architecture"])
            training = SimpleNamespace(**model_config["training"])

            # Create a SimpleNamespace object with all config variables
            return SimpleNamespace(
                batch_size=model_config["batch_size"],
                num_training_updates=model_config["num_training_updates"],
                architecture=architecture,
                training=training,
            )
        except yaml.YAMLError as e:
            print(f"Error reading the YAML file: {e}")
            return None


def main():
    config_path = "config.yaml"  # Adjust this path as needed
    config = read_config(config_path)
    if config:
        print("Model Configuration:")
        print(f"Batch Size: {config.batch_size}")
        print(f"Number of Training Updates: {config.num_training_updates}")
        print("\nArchitecture:")
        for key, value in vars(config.architecture).items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        print("\nTraining Parameters:")
        for key, value in vars(config.training).items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("Failed to read the configuration file.")

    data_path = "skiing_observations.npy"

    data = np.load(data_path)
    data = data.astype(np.float32) / 255.0  # convert data to float and norm

    print("\n--- Loading Data ---")
    print(f"Loaded Data: {data.shape[0]}")

    train_data, test_data = random_split(data, [0.8, 0.2])

    print(f"Train Data: {len(train_data)}")
    print(f"Test Data: {len(test_data)}")

    data_variance = np.var(data[train_data.indices])
    print(data[train_data.indices].shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # DataLoaders
    training_loader = DataLoader(
        train_data,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=True,
    )
    validataion_loader = DataLoader(
        test_data, batch_size=32, shuffle=True, pin_memory=True
    )

    # VQ-VAE Model
    model = VQVAE(
        config.architecture.num_hiddens,
        config.architecture.num_residual_layers,
        config.architecture.num_residual_hiddens,
        config.architecture.num_embeddings,
        config.architecture.embedding_dim,
        config.training.commitment_cost,
        config.training.decay,
    ).to(device)

    optimizer = optim.Adam(
        params=model.parameters(), lr=config.training.learning_rate, amsgrad=False
    )

    # TensorBoard setup
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    log_dir = os.path.join("runs", current_time)
    writer = SummaryWriter(log_dir)

    print("\n--- Training ---")
    # Training Loop
    pbar = tqdm(range(config.num_training_updates))
    for i in pbar:
        data = next(iter(training_loader))  # current batch of data
        data = torch.Tensor(data).to(device)
        # convert from B, H, W, C -> B, C, H, W
        data = data.permute(0, 3, 1, 2).contiguous()

        optimizer.zero_grad()

        if i == config.training.pretrain_steps:
            # K Means Cluster to initialize discrete embeddings
            with torch.no_grad():
                embeddings = model.encode(data)

            embeddings = (
                embeddings.permute(0, 2, 3, 1)
                .contiguous()
                .reshape(-1, config.architecture.embedding_dim)
            )

            np_e = embeddings.cpu().detach().numpy()

            n_clusters = config.architecture.num_embeddings
            kmeans = KMeans(n_clusters)
            kmeans.fit(np_e)

            cluster_centers = torch.from_numpy(
                kmeans.cluster_centers_).to(device)

            model.set_embeddings(cluster_centers)

        # Pretrain without vector quantizer
        if i < config.training.pretrain_steps:
            data_recon = model.pretrain(data)
            loss = nn.functional.mse_loss(data_recon, data) / data_variance
            recon_error = loss
            vq_loss = torch.Tensor([0.0])
            perplexity = torch.Tensor([0.0])
        else:
            vq_loss, data_recon, perplexity = model(data)
            recon_error = nn.functional.mse_loss(
                data_recon, data) / data_variance
            loss = recon_error + vq_loss

        loss.backward()
        optimizer.step()

        pbar.set_postfix(loss=loss.item())

        # Log to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), i)
        writer.add_scalar("VQ Loss", vq_loss.item(), i)
        writer.add_scalar("Reconstruction Error", recon_error.item(), i)
        writer.add_scalar("Perplexity", perplexity.item(), i)

        # Save model checkpoint every 1000 updates
        if (i + 1) % 1000 == 0:
            checkpoint_path = os.path.join(
                "checkpoints", f"model_checkpoint_{i+1}.pth")
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(
                {
                    "epoch": i + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": loss,
                },
                checkpoint_path,
            )
            pbar.write(f"Checkpoint saved at {checkpoint_path}")

    writer.close()
    print("Training completed.")


if __name__ == "__main__":
    main()
