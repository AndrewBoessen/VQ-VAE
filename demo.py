import matplotlib.pyplot as plt
import numpy as np
import torch
import umap.umap_ as umap

from gameplay_dataset_reader import GameFrameDataset
from train_tokenizer import EncoderDecoderConfig, load_config
from vqvae import Decoder, Encoder, Tokenizer


def load_checkpoint(model, checkpoint_path, device):
    """
    Load model paramters from checkpoint

    :param model nn.Module: model to load parameters for
    :param checkpoint_path str: file path to checkpoint
    :param device str: device to load model to
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def visualize_reconstructions(model, data, num_samples=5):
    """
    Generate image reconstructions with vq-vae decoder network

    :param model nn.Module: model to use for generating reconstructions
    :param data numpy.ndarray: image to resonstruct with network
    :param num_samples number: number of samples in data
    """
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(data), num_samples, replace=False)
        samples = torch.stack([data[i]["image"] for i in indices]).to(
            next(model.parameters()).device
        )

        # Reconstruct images
        _, _, reconstructions = model(
            samples, should_preprocess=True, should_postprocess=True
        )

        samples = (torch.clamp(samples, 0.0, 1.0) * 255.0).to(dtype=torch.uint8)
        reconstructions = (torch.clamp(reconstructions, 0.0, 1.0) * 255.0).to(
            dtype=torch.uint8
        )

        # Visualize original and reconstructed images
        _, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        for i in range(num_samples):
            axes[0, i].imshow(samples[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].axis("off")
            axes[0, i].set_title("Original")

            axes[1, i].imshow(reconstructions[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].axis("off")
            axes[1, i].set_title("Reconstructed")

        plt.tight_layout()
        plt.show()


def analyze_embeddings_umap(model, data, num_samples=1000, batch_size=32):
    """
    UMAP analysis for discrete embeddings

    :param model nn.Module: model embeddings to visualize
    :param data numpy.ndarray: data to use for sampling embeddings
    :param num_samples number: number of samples in data
    :param batch_size number: size of each batch for encoding
    """
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(data), num_samples, replace=False)

        # Initialize list to store embeddings
        all_embeddings = []

        # Process in batches
        for start_idx in range(0, num_samples, batch_size):
            end_idx = min(start_idx + batch_size, num_samples)
            batch_indices = indices[start_idx:end_idx]
            samples = torch.stack([data[i]["image"] for i in batch_indices]).to(
                next(model.parameters()).device
            )

            # Get embeddings for the current batch
            batch_embeddings = model.encode(samples)
            batch_embeddings = (
                batch_embeddings.z_quantized.view(len(batch_indices), -1).cpu().numpy()
            )

            # Append to the list of all embeddings
            all_embeddings.append(batch_embeddings)

        # Concatenate all batch embeddings into a single array
        embeddings = np.concatenate(all_embeddings, axis=0)

        # Perform UMAP
        umap_reducer = umap.UMAP(
            n_neighbors=3, min_dist=0.1, n_components=2, random_state=42
        )
        embeddings_2d = umap_reducer.fit_transform(embeddings)

        # Visualize UMAP results
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title("UMAP visualization of VQ-VAE embeddings")
        plt.show()


def main():
    # Load configuration
    config_path = "config_256.yaml"
    config = load_config(config_path)
    # Create encoder/decoder config from loaded configuration
    encoder_decoder_config = EncoderDecoderConfig(
        resolution=config["encoder"]["config"]["resolution"],
        in_channels=config["encoder"]["config"]["in_channels"],
        z_channels=config["encoder"]["config"]["z_channels"],
        ch=config["encoder"]["config"]["ch"],
        ch_mult=tuple(config["encoder"]["config"]["ch_mult"]),
        num_res_blocks=config["encoder"]["config"]["num_res_blocks"],
        attn_resolutions=tuple(config["encoder"]["config"]["attn_resolutions"]),
        out_ch=config["encoder"]["config"]["out_ch"],
        dropout=config["encoder"]["config"]["dropout"],
    )
    # Load data
    val_path = "gameplay_data_256_no_crop/val/"

    data = GameFrameDataset(val_path)

    # Initialize model components using config values
    encoder = Encoder(config=encoder_decoder_config)
    decoder = Decoder(config=encoder_decoder_config)
    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Initialize Tokenizer using config values
    model = Tokenizer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        encoder=encoder,
        decoder=decoder,
        with_lpips=True,
    ).to(device)

    # Load checkpoint
    checkpoint_path = "experiments/no_crop_res_256_attn/checkpoint_epoch_15.pt"  # Adjust path as needed
    model = load_checkpoint(model, checkpoint_path, device)

    print("Model loaded successfully.")

    # Visualize reconstructions
    visualize_reconstructions(model, data)

    # Analyze embeddings using UMAP
    analyze_embeddings_umap(model, data)


if __name__ == "__main__":
    main()
