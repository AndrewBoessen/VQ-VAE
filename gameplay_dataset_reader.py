import os
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


@dataclass
class PreprocessingConfig:
    resize_shorter_edge: int
    crop_size: int
    random_crop: bool
    random_flip: bool


class GameFrameDataset(Dataset):
    """
    Dataset class for loading video game frames and their corresponding actions.
    Handles sharded numpy files containing frame data and action labels.
    """

    def __init__(self, shard_dir: str, preload_shards: bool = False):
        """
        Initialize the dataset.

        Args:
            shard_dir (str): Directory containing the sharded numpy files
            transform: Optional transforms to apply to the images
            preload_shards (bool): If True, loads all shards into memory at init
        """
        self.shard_dir = shard_dir
        self.transform = [
            transforms.ToTensor(),
        ]
        self.preload_shards = preload_shards

        # Get all shard files
        self.frame_shards = sorted(
            [
                f
                for f in os.listdir(shard_dir)
                if f.startswith("frames_") and f.endswith(".npy")
            ]
        )
        self.action_shards = sorted(
            [
                f
                for f in os.listdir(shard_dir)
                if f.startswith("actions_") and f.endswith(".npy")
            ]
        )

        # Verify matching shards
        assert len(self.frame_shards) == len(
            self.action_shards
        ), "Number of frame and action shards must match"

        # Load shard metadata to get total dataset size
        self.shard_sizes = []
        self.cumulative_sizes = [0]

        for frame_shard in self.frame_shards:
            shard_path = os.path.join(shard_dir, frame_shard)
            # Load just the shape information without loading the full array
            shard_size = np.load(shard_path, mmap_mode="r").shape[0]
            self.shard_sizes.append(shard_size)
            self.cumulative_sizes.append(self.cumulative_sizes[-1] + shard_size)

        # Initialize shard cache if preloading
        self.shard_cache = {}
        if preload_shards:
            self._preload_all_shards()

    def _preload_all_shards(self):
        """Preload all shards into memory."""
        for idx in range(len(self.frame_shards)):
            frame_path = os.path.join(self.shard_dir, self.frame_shards[idx])
            action_path = os.path.join(self.shard_dir, self.action_shards[idx])

            self.shard_cache[idx] = {
                "frames": np.load(frame_path),
                "actions": np.load(action_path),
            }

    def _load_shard(self, shard_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load a specific shard.

        Args:
            shard_idx (int): Index of the shard to load

        Returns:
            Tuple containing the frames and actions arrays
        """
        if self.preload_shards:
            return (
                self.shard_cache[shard_idx]["frames"],
                self.shard_cache[shard_idx]["actions"],
            )

        frame_path = os.path.join(self.shard_dir, self.frame_shards[shard_idx])
        action_path = os.path.join(self.shard_dir, self.action_shards[shard_idx])

        frames = np.load(frame_path)
        actions = np.load(action_path)

        return frames, actions

    def _get_shard_and_idx(self, idx: int) -> Tuple[int, int]:
        """
        Convert global index to shard index and local index.

        Args:
            idx (int): Global index into the dataset

        Returns:
            Tuple of (shard_idx, local_idx)
        """
        shard_idx = (
            next(
                i for i, cum_size in enumerate(self.cumulative_sizes) if cum_size > idx
            )
            - 1
        )
        local_idx = idx - self.cumulative_sizes[shard_idx]
        return shard_idx, local_idx

    def __len__(self) -> int:
        """Return total size of the dataset."""
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single item from the dataset.

        Args:
            idx (int): Index of the item to get

        Returns:
            Dictionary of (frame, action) tensors
        """
        if idx >= len(self):
            raise IndexError("Index out of bounds")

        # Get shard and local index
        shard_idx, local_idx = self._get_shard_and_idx(idx)

        # Load shard data
        frames, actions = self._load_shard(shard_idx)

        # Get specific frame and action
        frame = frames[local_idx]
        action = actions[local_idx]

        # Convert frame to PIL Image for transforms
        frame = Image.fromarray(frame)

        # Apply transforms if specified
        if self.transform:
            transforms_list = transforms.Compose(self.transform)
            frame = transforms_list(frame)
        else:
            frame = torch.from_numpy(np.array(frame)).permute(2, 0, 1)

        action = torch.tensor(action)

        return {"image": frame, "action": action}
