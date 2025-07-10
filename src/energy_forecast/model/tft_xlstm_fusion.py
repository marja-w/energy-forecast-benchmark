import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from dataclasses import dataclass
import math

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
class TFTxLSTMConfig:
    """Configuration for TFT-style xLSTM with LSTM encoders."""
    # Model dimensions
    embedding_dim: int
    hidden_dim: int
    context_length: int
    future_length: int

    # Feature dimensions
    past_feature_dim: int
    future_feature_dim: int
    static_feature_dim: int = 0

    # LSTM encoder settings
    encoder_layers: int = 2
    encoder_dropout: float = 0.1

    # xLSTM settings
    xlstm_layers: int = 4

    # Fusion settings
    fusion_method: str = "gated_attention"  # "attention", "gated_attention", "concat"
    num_attention_heads: int = 8

    # General settings
    bias: bool = True
    dropout: float = 0.1
    add_post_blocks_norm: bool = True

    device: str = "cpu"

    def get(self, key: str, default: Any = None):
        return getattr(self, key, default)


class VariableSelectionNetwork(nn.Module):
    """Fixed Variable selection network for feature importance."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.1,
            context_dim: int = 0
    ):
        super().__init__()
        self.input_dim = input_dim  # Number of input features (21 in your case)
        self.hidden_dim = hidden_dim
        self.context_dim = context_dim

        # GRN to compute variable selection weights
        self.variable_selection_grn = GatedResidualNetwork(
            input_dim=input_dim + context_dim,  # All features + optional context
            hidden_dim=hidden_dim,
            output_dim=input_dim,  # Output weight for each input feature
            dropout=dropout
        )

        # Individual GRNs for each variable
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(
                input_dim=1 + context_dim,  # Single feature + optional context
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            for _ in range(input_dim)
        ])

        self.softmax = nn.Softmax(dim=-1)

    def forward(
            self,
            x: torch.Tensor,
            context: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - your 21 features
            context: [batch_size, context_dim] - optional static context

        Returns:
            selected_features: [batch_size, seq_len, hidden_dim]
            weights: [batch_size, seq_len, input_dim] - attention weights for each feature
        """
        batch_size, seq_len, num_features = x.shape

        # Prepare input for variable selection
        if context is not None:
            # Broadcast context to all timesteps
            context_expanded = context.unsqueeze(1).expand(-1, seq_len, -1)
            selection_input = torch.cat([x, context_expanded], dim=-1)
        else:
            selection_input = x

        # Flatten for processing
        selection_input_flat = selection_input.view(-1, selection_input.size(-1))

        # Compute variable selection weights using the correctly named GRN
        weights_flat = self.variable_selection_grn(selection_input_flat)
        weights = self.softmax(weights_flat).view(batch_size, seq_len, num_features)

        # Process each variable individually
        variable_outputs = []
        for i in range(num_features):
            # Extract single feature
            single_var = x[:, :, i:i + 1]  # [batch_size, seq_len, 1]

            # Prepare input for individual GRN
            if context is not None:
                var_input = torch.cat([single_var, context_expanded], dim=-1)
            else:
                var_input = single_var

            # Process through individual GRN
            var_input_flat = var_input.view(-1, var_input.size(-1))
            var_output_flat = self.variable_grns[i](var_input_flat)
            var_output = var_output_flat.view(batch_size, seq_len, self.hidden_dim)

            variable_outputs.append(var_output)

        # Stack and weight
        stacked_outputs = torch.stack(variable_outputs, dim=-1)  # [B, T, H, num_features]
        selected_features = torch.sum(
            stacked_outputs * weights.unsqueeze(-2),
            dim=-1
        )  # [B, T, H]

        return selected_features, weights


class EfficientVariableSelectionNetwork(nn.Module):
    """Most efficient VSN implementation."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # Compute attention weights for each feature
        self.attention_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
            nn.Softmax(dim=-1)
        )

        # Process features
        self.feature_processor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim] - your 21 features

        Returns:
            selected_features: [batch_size, seq_len, hidden_dim]
            weights: [batch_size, seq_len, input_dim]
        """
        # Compute attention weights
        weights = self.attention_layer(x)  # [B, T, input_dim]

        # Apply attention to input
        attended_input = x * weights  # [B, T, input_dim]

        # Process attended features
        selected_features = self.feature_processor(attended_input)  # [B, T, hidden_dim]

        return selected_features, weights


class GatedResidualNetwork(nn.Module):
    """Gated Residual Network from TFT."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: Optional[int] = None,
            dropout: float = 0.1,
            use_static_context: bool = True
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim or input_dim
        self.use_static_context = use_static_context

        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.gate_linear = nn.Linear(hidden_dim, self.output_dim)

        if self.output_dim != input_dim:
            self.skip_linear = nn.Linear(input_dim, self.output_dim)
        else:
            self.skip_linear = nn.Identity()

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(self.output_dim)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor, static_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: [batch_size, ..., input_dim] or [batch_size * seq_len, input_dim]
            static_context: [batch_size, static_dim] - not used in this simplified version
        """
        # Store original shape for reshaping
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.view(-1, original_shape[-1])

        # Skip connection
        skip = self.skip_linear(x)

        # Main path
        hidden = self.elu(self.linear1(x))
        hidden = self.dropout(hidden)
        hidden = self.linear2(hidden)

        # Gating
        gate = self.sigmoid(self.gate_linear(hidden))
        output = gate * hidden + skip

        # Layer norm
        output = self.layer_norm(output)

        # Reshape back to original
        if len(original_shape) > 2:
            output = output.view(*original_shape[:-1], -1)

        return output


class LSTMEncoder(nn.Module):
    """LSTM encoder for past or future sequences."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 2,
            dropout: float = 0.1,
            bidirectional: bool = False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)

        # Output dimension adjustment for bidirectional
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [batch_size, seq_len, input_dim]
            mask: [batch_size, seq_len] - optional padding mask

        Returns:
            output: [batch_size, seq_len, hidden_dim * directions]
            final_state: [batch_size, hidden_dim * directions]
        """
        if mask is not None:
            # Pack padded sequence for efficiency
            lengths = mask.sum(dim=1).cpu()
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )

        output, (h_n, c_n) = self.lstm(x)

        if mask is not None:
            # Unpack sequence
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = self.dropout(output)

        # Get final state (concatenate forward and backward if bidirectional)
        if self.bidirectional:
            final_state = torch.cat([h_n[-2], h_n[-1]], dim=-1)  # [batch_size, hidden_dim * 2]
        else:
            final_state = h_n[-1]  # [batch_size, hidden_dim]

        return output, final_state


class ContextFusion(nn.Module):
    """Fuses past and future contexts using attention or gating."""

    def __init__(
            self,
            past_dim: int,
            future_dim: int,
            hidden_dim: int,
            fusion_method: str = "gated_attention",
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        super().__init__()
        self.fusion_method = fusion_method
        self.hidden_dim = hidden_dim

        if fusion_method == "attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.norm = nn.LayerNorm(hidden_dim)

        elif fusion_method == "gated_attention":
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            self.gate_network = GatedResidualNetwork(
                input_dim=hidden_dim * 2,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
                dropout=dropout
            )
            self.norm = nn.LayerNorm(hidden_dim)

        elif fusion_method == "concat":
            self.fusion_linear = nn.Linear(past_dim + future_dim, hidden_dim)

        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")

        # Ensure input dimensions match hidden_dim
        if past_dim != hidden_dim:
            self.past_proj = nn.Linear(past_dim, hidden_dim)
        else:
            self.past_proj = nn.Identity()

        if future_dim != hidden_dim:
            self.future_proj = nn.Linear(future_dim, hidden_dim)
        else:
            self.future_proj = nn.Identity()

    def forward(
            self,
            past_context: torch.Tensor,
            future_context: torch.Tensor,
            past_mask: Optional[torch.Tensor] = None,
            future_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            past_context: [batch_size, past_len, past_dim]
            future_context: [batch_size, future_len, future_dim]
            past_mask: [batch_size, past_len]
            future_mask: [batch_size, future_len]

        Returns:
            fused_context: [batch_size, past_len, hidden_dim]
        """
        # Project to common dimension
        past_proj = self.past_proj(past_context)
        future_proj = self.future_proj(future_context)

        if self.fusion_method == "attention":
            # Past attends to future
            attended, _ = self.cross_attention(
                query=past_proj,
                key=future_proj,
                value=future_proj,
                key_padding_mask=~future_mask if future_mask is not None else None
            )
            fused = self.norm(attended + past_proj)

        elif self.fusion_method == "gated_attention":
            # Cross-attention
            attended, _ = self.cross_attention(
                query=past_proj,
                key=future_proj,
                value=future_proj,
                key_padding_mask=~future_mask if future_mask is not None else None
            )

            # Gated fusion
            concatenated = torch.cat([past_proj, attended], dim=-1)
            gated = self.gate_network(concatenated)
            fused = self.norm(gated + past_proj)

        elif self.fusion_method == "concat":
            # Broadcast future to past length
            batch_size, past_len, _ = past_context.shape
            future_mean = future_proj.mean(dim=1, keepdim=True)  # [B, 1, H]
            future_broadcast = future_mean.expand(-1, past_len, -1)  # [B, T, H]

            concatenated = torch.cat([past_proj, future_broadcast], dim=-1)
            fused = self.fusion_linear(concatenated)

        return fused


class TFTxLSTMModel(nn.Module):
    """TFT-style xLSTM with LSTM encoders for past and future."""

    def __init__(
            self,
            out_features: int,
            config: TFTxLSTMConfig
    ):
        super().__init__()
        self.config = config

        # Variable selection networks
        if config.past_feature_dim > 0:
            self.past_vsn = EfficientVariableSelectionNetwork(
                input_dim=config.past_feature_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout
            )

        if config.future_feature_dim > 0:
            self.future_vsn = EfficientVariableSelectionNetwork(
                input_dim=config.future_feature_dim,
                hidden_dim=config.hidden_dim,
                dropout=config.dropout
            )

        # LSTM Encoders
        self.past_encoder = LSTMEncoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            dropout=config.encoder_dropout,
            bidirectional=False
        )

        self.future_encoder = LSTMEncoder(
            input_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.encoder_layers,
            dropout=config.encoder_dropout,
            bidirectional=False
        )

        # Context fusion
        self.context_fusion = ContextFusion(
            past_dim=config.hidden_dim,
            future_dim=config.hidden_dim,
            hidden_dim=config.embedding_dim,
            fusion_method=config.fusion_method,
            num_heads=config.num_attention_heads,
            dropout=config.dropout
        )

        # Static covariate processing
        if config.static_feature_dim > 0:
            self.static_encoder = GatedResidualNetwork(
                input_dim=config.static_feature_dim,
                hidden_dim=config.hidden_dim,
                output_dim=config.embedding_dim,
                dropout=config.dropout
            )

        # xLSTM blocks (placeholder - use your implementation)
        self.blocks = self._create_xlstm_blocks(config)

        # Post-processing
        if config.add_post_blocks_norm:
            self.post_blocks_norm = nn.LayerNorm(config.embedding_dim)
        else:
            self.post_blocks_norm = nn.Identity()

        # Output projection
        self.flatten = nn.Flatten()
        self.proj_down = nn.Linear(
            in_features=config.embedding_dim * config.context_length,
            out_features=out_features,
            bias=config.bias
        )

        # Positional encoding
        self.positional_encoding = nn.Parameter(
            torch.randn(config.context_length, config.embedding_dim) * 0.02
        )

        self.dropout = nn.Dropout(config.dropout)

    def _create_xlstm_blocks(self, config: TFTxLSTMConfig):
        """Create xLSTM blocks - replace with your implementation."""
        # Placeholder - use your existing xLSTM block creation logic
        backend = "vanilla" if config.device.type == "cpu" else "cuda"
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=1  # TODO: fails for > 1
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=1,  # TODO: fails for > 1
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=config.get("hidden_dim", 256),  # TODO: n_in is context_length?
            num_blocks=self.config.get("num_blocks", 7),
            embedding_dim=config.get("embedding_dim", 256),  # TODO: number of features is embedding dimension?
            slstm_at=[1],
        )
        return xLSTMBlockStack(cfg)

    def forward(
            self,
            past_features: torch.Tensor,
            future_features: Optional[torch.Tensor] = None,
            static_features: Optional[torch.Tensor] = None,
            past_mask: Optional[torch.Tensor] = None,
            future_mask: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            past_features: [batch_size, context_length, past_feature_dim]
            future_features: [batch_size, future_length, future_feature_dim]
            static_features: [batch_size, static_feature_dim]
            past_mask: [batch_size, context_length]
            future_mask: [batch_size, future_length]

        Returns:
            Dictionary with predictions and attention weights
        """
        # Variable selection for past features
        past_selected, past_weights = self.past_vsn(past_features)

        # Encode past context
        past_encoded, past_final = self.past_encoder(past_selected, past_mask)

        # Process future features if available
        if future_features is not None and self.config.future_feature_dim > 0:
            future_selected, future_weights = self.future_vsn(future_features)
            future_encoded, future_final = self.future_encoder(future_selected, future_mask)

            # Fuse contexts
            fused_context = self.context_fusion(
                past_context=past_encoded,
                future_context=future_encoded,
                past_mask=past_mask,
                future_mask=future_mask
            )
        else:
            fused_context = past_encoded
            if fused_context.size(-1) != self.config.embedding_dim:
                projection = nn.Linear(
                    fused_context.size(-1),
                    self.config.embedding_dim
                ).to(fused_context.device)
                fused_context = projection(fused_context)
            future_weights = None
            future_encoded = None

        # Add positional encoding
        fused_context = fused_context + self.positional_encoding.unsqueeze(0)
        fused_context = self.dropout(fused_context)

        # Pass through xLSTM blocks
        x = fused_context
        x = self.blocks(x)

        # Post-processing
        # x = self.post_blocks_norm(x)
        x = self.flatten(x)
        predictions = self.proj_down(x)

        return {
            'predictions': predictions,
            'past_weights': past_weights,
            'future_weights': future_weights,
            'past_encoded': past_encoded,
            'future_encoded': future_encoded,
            'fused_context': fused_context
        }


# Usage example
def create_tft_xlstm_model():
    """Example of creating and using the TFT-style xLSTM model."""
    config = TFTxLSTMConfig(
        embedding_dim=256,
        hidden_dim=256,
        context_length=100,
        future_length=20,
        past_feature_dim=50,
        future_feature_dim=10,
        static_feature_dim=5,
        encoder_layers=2,
        xlstm_layers=4,
        fusion_method="gated_attention",
        num_attention_heads=8,
        dropout=0.1
    )

    model = TFTxLSTMModel(
        out_features=1,
        config=config
    )

    return model, config
