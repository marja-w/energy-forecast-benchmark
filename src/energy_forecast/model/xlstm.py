from dataclasses import dataclass
from typing import Any

import torch
from loguru import logger

from src.energy_forecast.xlstm.xlstm import xLSTMBlockStackConfig
from torch import nn

from src.energy_forecast.xlstm.xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

@dataclass
class xLSTMConfig:
    """Configuration for adapted xLSTM"""
    # Feature sizes
    in_features: int
    out_features: int
    context_length: int

    # Model dimensions
    embedding_dim: int
    hidden_dim: int
    num_blocks: int
    num_heads: int = 4

    # Device settings
    device: str = "cpu"

    # General settings
    bias: bool = True
    dropout: float = 0.1
    add_post_blocks_norm: bool = True

    def get(self, key: str, default: Any = None):
        return getattr(self, key, default)

class xLSTMAdaptModel(nn.Module):
    """xLSTM adaptation for time-series forecasting use case with original xLSTM implementation"""

    def __init__(self, config: xLSTMConfig):
        super().__init__()
        self.config = config

        # up-projection layer
        self.proj_up = nn.Linear(
            in_features=self.config.in_features,  # number of input features
            out_features=self.config.embedding_dim,  # input dimension of blocks
            bias=self.config.bias
        )

        # xLSTM blocks (official implementation)
        self.blocks = self._create_xlstm_blocks()

        # down-projection layer
        self.flatten = nn.Flatten()
        self.proj_down = nn.Linear(
            in_features=self.config.embedding_dim*self.config.context_length,
            out_features=self.config.out_features,
            bias=self.config.bias
        )

    def _create_xlstm_blocks(self):
        """Create xLSTM blocks - replace with your implementation."""
        # Placeholder - use your existing xLSTM block creation logic
        backend = "vanilla" if self.config.device.type == "cpu" else "cuda"
        logger.info(f"Using backend: {backend}")
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=self.config.num_heads  # TODO: fails for > 1
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=self.config.num_heads,  # TODO: fails for > 1
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=self.config.get("context_length", 7),
            num_blocks=self.config.get("num_blocks", 7),
            embedding_dim=self.config.get("embedding_dim", 256),
            slstm_at=[1],
        )
        return xLSTMBlockStack(cfg)

    def forward(self, x: torch.Tensor, **kwargs):
        x = self.proj_up(x)

        # pass through xLSTM
        x = self.blocks(x)

        x = self.flatten(x)
        x = self.proj_down(x)
        return x