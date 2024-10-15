import os
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from types import SimpleNamespace

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
                training=training
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

    print(f"Loaded Data: {data.shape[0]}")

    train_data, test_data = random_split(data, [0.8, 0.2])

    print(f"Train Data: {len(train_data)}")
    print(f"Test Data: {len(test_data)}")
    data_variance = np.var(data[train_data.indices])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # DataLoaders
    training_loader = DataLoader(
        train_data, batch_size=config.architecture.num_hiddens, shuffle=True, pin_memory=True)
    validataion_loader = DataLoader(
        test_data, batch_size=32, shuffle=True, pin_memory=True)

    # VQ-VAE Model
    model = VQVAE(config.architecture.num_hiddens, config.architecture.num_residual_layers, config.architecture.num_residual_hiddens,
                  config.architecture.num_embeddings, config.architecture.embedding_dim, config.training.commitment_cost, config.training.decay).to(device)

    optimizer = optim.Adam(params=model.parameters(),
                           lr=config.training.learning_rate, amsgrad=False)


if __name__ == "__main__":
    main()
