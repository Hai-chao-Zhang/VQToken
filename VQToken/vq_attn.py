"""Cross-attention used by the optional learned VQToken compression path."""

from __future__ import annotations

import torch
import torch.nn as nn


class VQAttn(nn.Module):
    def __init__(self, query_dim: int, context_dim: int, num_heads: int = 1, num_layers: int = 1):
        super().__init__()
        if query_dim < 1 or context_dim < 1:
            raise ValueError("query_dim and context_dim must be positive")
        if context_dim % num_heads != 0:
            raise ValueError("context_dim must be divisible by num_heads")

        self.query_dim = query_dim
        self.context_dim = context_dim
        self.to_q_proj = nn.Linear(query_dim, context_dim)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=context_dim,
            nhead=num_heads,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        # Keep normalization scales at one. The previous implementation reset
        # them to values near zero, which effectively suppressed fresh models.
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                if module.weight is not None:
                    nn.init.ones_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Attend from ``[B, Lq, query_dim]`` to ``[B, Lc, context_dim]``."""

        if x.ndim != 3 or context.ndim != 3:
            raise ValueError("x and context must both be 3D tensors")
        if x.shape[0] != context.shape[0]:
            raise ValueError("x and context must have the same batch size")
        if x.shape[-1] != self.query_dim or context.shape[-1] != self.context_dim:
            raise ValueError(
                f"expected trailing dimensions {self.query_dim} and {self.context_dim}, "
                f"got {x.shape[-1]} and {context.shape[-1]}"
            )
        if x.shape[1] == 0 or context.shape[1] == 0:
            raise ValueError("x and context sequences must be non-empty")

        projected_query = self.to_q_proj(x)
        return self.transformer_decoder(tgt=projected_query, memory=context)

    def cross_attention_weighted_clusters(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """Cross-attend assignment-map features to a discrete codebook.

        Two-dimensional inputs represent unbatched sequences and are promoted
        to batch size one. A batched tensor can be paired with an unbatched
        tensor, which is expanded across the batch. The shorter of the
        assignment-map and codebook sequences is then repeated to the longer
        sequence length. This preserves the alignment used to train the
        released checkpoint. With the supported deployment invariant
        ``num_frames <= codebook_size``, the output has one token per codebook
        entry.
        """

        if x.ndim not in (2, 3) or context.ndim not in (2, 3):
            raise ValueError("x and context must be 2D or 3D tensors")
        squeeze_batch = x.ndim == 2 and context.ndim == 2

        if x.ndim == 2:
            x = x.unsqueeze(0)
        if context.ndim == 2:
            context = context.unsqueeze(0)

        if x.shape[0] != context.shape[0]:
            if x.shape[0] == 1:
                x = x.expand(context.shape[0], -1, -1)
            elif context.shape[0] == 1:
                context = context.expand(x.shape[0], -1, -1)
            else:
                raise ValueError("x and context batch sizes must match, or one must be 1")

        query_length, codebook_length = x.shape[1], context.shape[1]
        if query_length == 0 or codebook_length == 0:
            raise ValueError("x and context sequences must be non-empty")
        target_length = max(query_length, codebook_length)
        if query_length < target_length:
            repeats = (target_length + query_length - 1) // query_length
            x = x.repeat(1, repeats, 1)[:, :target_length, :]
        if codebook_length < target_length:
            repeats = (target_length + codebook_length - 1) // codebook_length
            context = context.repeat(1, repeats, 1)[:, :target_length, :]

        output = self.forward(x, context)
        return output.squeeze(0) if squeeze_batch else output
