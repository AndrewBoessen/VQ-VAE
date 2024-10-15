import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
from datetime import datetime

def read_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config['model_config']
        except yaml.YAMLError as e:
            print(f"Error reading the YAML file: {e}")
            return None

def main():
    config_path = 'config.yaml'  # Adjust this path as needed
    config = read_config(config_path)

    if config:
        print("Model Configuration:")
        print(f"Batch Size: {config['batch_size']}")
        print(f"Number of Training Updates: {config['num_training_updates']}")
        
        print("\nArchitecture:")
        for key, value in config['architecture'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        print("\nTraining Parameters:")
        for key, value in config['training'].items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
    else:
        print("Failed to read the configuration file.")

if __name__ == "__main__":
    main()
