import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from gameplay_dataset_reader import GameFrameDataset, PreprocessingConfig
from vqvae import Decoder, Encoder, Tokenizer


@dataclass
class EncoderDecoderConfig:
    resolution: int
    in_channels: int
    z_channels: int
    ch: int
    ch_mult: tuple
    num_res_blocks: int
    attn_resolutions: tuple
    out_ch: int
    dropout: float


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and process the YAML config file"""
    with open(config_path, "r") as f:
        # Load YAML with OmegaConf to handle the ${} references
        config = OmegaConf.load(f)

    # Convert to dictionary
    config = OmegaConf.to_container(config, resolve=True)

    # Add training specific parameters
    training_config = {
        "batch_size": 32,
        "num_workers": 4,
        "learning_rate": 1e-4,
        "min_lr": 1e-6,
        "beta1": 0.9,
        "beta2": 0.999,
        "num_epochs": 100,
        "log_every": 100,
    }

    # Merge configs
    config.update(training_config)

    return config


class VQVAETrainer:
    def __init__(
        self,
        config: Dict[str, Any],
        model: Tokenizer,
        train_dataset: GameFrameDataset,
        val_dataset: GameFrameDataset,
        device: str = "cuda",
    ):
        self.config = config
        self.model = model.to(device)
        self.device = device
        self.global_step = 0

        # Initialize dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            num_workers=config["num_workers"],
            pin_memory=True,
        )

        # Setup optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config["learning_rate"],
            betas=(config["beta1"], config["beta2"]),
        )

        # Setup learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config["num_epochs"], eta_min=config["min_lr"]
        )

        # Initialize logging
        self.setup_logging()

    def setup_logging(self):
        # Create experiment directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = os.path.join("checkpoints", f"vqvae_{timestamp}")
        os.makedirs(self.exp_dir, exist_ok=True)

        # Setup file logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(os.path.join(self.exp_dir, "training.log")),
                logging.StreamHandler(),
            ],
        )
        self.logger = logging.getLogger(__name__)

        # Setup tensorboard
        self.writer = SummaryWriter(log_dir=self.exp_dir)

    def save_checkpoint(self, epoch: int):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "global_step": self.global_step,
            "config": self.config,
        }
        path = os.path.join(self.exp_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(checkpoint, path)
        self.logger.info(f"Saved checkpoint: {path}")

    def load_checkpoint(self, checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.global_step = checkpoint["global_step"]
        return checkpoint["epoch"]

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str = "train"):
        """Log metrics to both tensorboard and logging file"""
        for name, value in metrics.items():
            self.writer.add_scalar(f"{prefix}/{name}", value, step)

        metrics_str = " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.logger.info(f"{prefix} step {step}: {metrics_str}")

    def log_images(
        self,
        batch: torch.Tensor,
        reconstructions: torch.Tensor,
        step: int,
        prefix: str = "train",
    ):
        """Log original and reconstructed images to tensorboard"""
        # Take the first few images from the batch
        num_images = min(4, batch.size(0))
        comparison = torch.cat([batch[:num_images], reconstructions[:num_images]])

        self.writer.add_images(
            f"{prefix}/reconstructions", comparison, step, dataformats="NCHW"
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        epoch_losses = []

        with tqdm(self.train_loader, desc=f"Epoch {epoch}") as pbar:
            for batch_idx, batch in enumerate(pbar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # Forward pass
                z, z_quantized, reconstructions = self.model(batch["observations"])
                loss = self.model.compute_loss(batch)
                total_loss = (
                    loss.commitment_loss
                    + loss.reconstruction_loss
                    + loss.perceptual_loss
                )

                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                # Log metrics
                metrics = {
                    "total_loss": total_loss.item(),
                    "commitment_loss": loss.commitment_loss.item(),
                    "reconstruction_loss": loss.reconstruction_loss.item(),
                    "perceptual_loss": loss.perceptual_loss.item(),
                }
                epoch_losses.append(total_loss.item())

                if batch_idx % self.config["log_every"] == 0:
                    self.log_metrics(metrics, self.global_step)
                    self.log_images(
                        batch["observations"], reconstructions, self.global_step
                    )

                pbar.set_postfix(loss=f"{total_loss.item():.4f}")
                self.global_step += 1

        return sum(epoch_losses) / len(epoch_losses)

    @torch.no_grad()
    def validate(self, epoch: int):
        self.model.eval()
        val_losses = []

        for batch in tqdm(self.val_loader, desc="Validation"):
            batch = {k: v.to(self.device) for k, v in batch.items()}

            # Forward pass
            z, z_quantized, reconstructions = self.model(batch["observations"])
            loss = self.model.compute_loss(batch)
            total_loss = (
                loss.commitment_loss + loss.reconstruction_loss + loss.perceptual_loss
            )

            val_losses.append(total_loss.item())

        avg_loss = sum(val_losses) / len(val_losses)
        metrics = {"val_loss": avg_loss}
        self.log_metrics(metrics, self.global_step, prefix="val")

        # Log validation reconstructions
        self.log_images(
            batch["observations"], reconstructions, self.global_step, prefix="val"
        )

        return avg_loss

    def train(self):
        self.logger.info("Starting training...")
        self.logger.info(f"Config: {self.config}")

        best_val_loss = float("inf")

        for epoch in range(self.config["num_epochs"]):
            train_loss = self.train_epoch(epoch)
            val_loss = self.validate(epoch)
            self.scheduler.step()

            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch)

            self.logger.info(
                f"Epoch {epoch} - Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, "
                f"LR: {self.scheduler.get_last_lr()[0]:.6f}"
            )


def main():
    # Load configuration from YAML
    config = load_config("config.yaml")

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

    # Initialize datasets
    train_dataset = GameFrameDataset(
        shard_dir="gameplay_data/train/", preload_shards=True
    )
    val_dataset = GameFrameDataset(shard_dir="gameplay_data/val/", preload_shards=True)

    # Initialize model components using config values
    encoder = Encoder(config=encoder_decoder_config)
    decoder = Decoder(config=encoder_decoder_config)

    # Initialize Tokenizer using config values
    model = Tokenizer(
        vocab_size=config["vocab_size"],
        embed_dim=config["embed_dim"],
        encoder=encoder,
        decoder=decoder,
        with_lpips=True,
    )

    # Initialize trainer
    trainer = VQVAETrainer(
        config=config,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # Start training
    trainer.train()


if __name__ == "__main__":
    main()
