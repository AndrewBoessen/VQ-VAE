import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import umap.umap_ as umap
from vqvae import VQVAE
from train import read_config


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model


def generate_samples(model, num_samples=10):
    model.eval()
    with torch.no_grad():
        # Generate random latent vectors
        latent_vectors = torch.randn(
            num_samples, model.embedding_dim, 8, 8)  # Adjust size as needed
        # Decode the latent vectors
        generated_images = model.decode(latent_vectors)
    return generated_images.cpu().numpy()


def visualize_reconstructions(model, data, num_samples=5):
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(data), num_samples, replace=False)
        samples = torch.Tensor(data[indices]).to(
            next(model.parameters()).device)
        samples = samples.permute(0, 3, 1, 2).contiguous()

        # Reconstruct images
        _, reconstructions, _ = model(samples)

        # Visualize original and reconstructed images
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        for i in range(num_samples):
            axes[0, i].imshow(samples[i].permute(1, 2, 0).cpu().numpy())
            axes[0, i].axis('off')
            axes[0, i].set_title('Original')

            axes[1, i].imshow(
                reconstructions[i].permute(1, 2, 0).cpu().numpy())
            axes[1, i].axis('off')
            axes[1, i].set_title('Reconstructed')

        plt.tight_layout()
        plt.show()


def analyze_embeddings_umap(model, data, num_samples=1000):
    model.eval()
    with torch.no_grad():
        # Select random samples
        indices = np.random.choice(len(data), num_samples, replace=False)
        samples = torch.Tensor(data[indices]).to(
            next(model.parameters()).device)
        samples = samples.permute(0, 3, 1, 2).contiguous()

        # Get embeddings
        embeddings = model.encode(samples)
        embeddings = embeddings.view(num_samples, -1).cpu().numpy()

        # Perform UMAP
        umap_reducer = umap.UMAP(n_neighbors=15, min_dist=0.1,
                                 n_components=2, random_state=42)
        embeddings_2d = umap_reducer.fit_transform(embeddings)

        # Visualize UMAP results
        plt.figure(figsize=(10, 8))
        plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5)
        plt.title('UMAP visualization of VQ-VAE embeddings')
        plt.colorbar()
        plt.show()


def main():
    # Load configuration
    config_path = "config.yaml"
    config = read_config(config_path)

    # Load data
    data_path = "skiing_observations.npy"
    data = np.load(data_path)
    data = data.astype(np.float32) / 255.0

    # Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VQVAE(
        config.architecture.num_hiddens,
        config.architecture.num_residual_layers,
        config.architecture.num_residual_hiddens,
        config.architecture.num_embeddings,
        config.architecture.embedding_dim,
        config.training.commitment_cost,
        config.training.decay,
    ).to(device)

    # Load checkpoint
    checkpoint_path = "checkpoints/model_checkpoint_20000.pth"  # Adjust path as needed
    model = load_checkpoint(model, checkpoint_path)

    print("Model loaded successfully.")

    # Generate and display samples
    generated_samples = generate_samples(model)
    plt.figure(figsize=(15, 3))
    for i in range(5):
        plt.subplot(1, 5, i+1)
        plt.imshow(generated_samples[i].transpose(1, 2, 0))
        plt.axis('off')
    plt.suptitle("Generated Samples")
    plt.show()

    # Visualize reconstructions
    visualize_reconstructions(model, data)

    # Analyze embeddings using UMAP
    analyze_embeddings_umap(model, data)


if __name__ == "__main__":
    main()
