import torch
import torch.nn as nn
import torch.nn.functional as F
# from VQToken.vq_token import kmeans_clustering_tokens_torch




class VQAttn(nn.Module):
    def __init__(self, query_dim, context_dim, num_heads=1, num_layers=1):
        super(VQAttn, self).__init__()
        self.query_dim = query_dim
        self.context_dim = context_dim
        
        # Transformer cross-attention setup
        decoder_layer = nn.TransformerDecoderLayer(d_model=context_dim, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        # Linear layer to match the input dimension of the query
        self.to_q_proj = nn.Linear(query_dim, context_dim)
        
        # Initialize weights
        self.initialize_weights()

    def initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight is not None and module.weight.numel() > 0:
                    nn.init.xavier_uniform_(module.weight)
                if module.bias is not None and module.bias.numel() > 0:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.TransformerDecoder):
                for name, param in module.named_parameters():
                    if param.numel() > 0:  # Avoid zero-dimensional parameters
                        if param.dim() > 1:
                            nn.init.xavier_uniform_(param)
                        elif param.dim() == 1 and 'bias' in name:
                            nn.init.zeros_(param)
                        elif param.dim() == 1:
                            nn.init.uniform_(param, -0.01, 0.01)

    def forward(self, x, context):
        # Project x to match context dimensions and permute for transformer requirements
        if x.numel() == 0 or context.numel() == 0:
            raise ValueError("Zero-sized tensor detected in forward pass.")

        x = self.to_q_proj(x).permute(1, 0, 2)  # [seq_len_q, batch_size, context_dim]
        context = context.permute(1, 0, 2)      # [seq_len_kv, batch_size, context_dim]

        # Perform transformer-based cross-attention
        try:
            output = self.transformer_decoder(tgt=x, memory=context)
        except AssertionError as e:
            print(f"Error in TransformerDecoder: {e}")
            raise

        return output.permute(1, 0, 2)  # Reshape back to [batch_size, seq_len_q, query_dim]

    def cross_attention_weighted_clusters(self, x, context):
        """
        Cross-attend from token indices to a discrete codebook and return
        attention-weighted code vectors.

        Args:
            x (torch.Tensor):
                Discrete token indices (integer dtype, e.g., torch.long).
                Shape: [B, Lq] or [Lq] (interpreted as B=1).
            context (torch.Tensor):
                Discrete token codebook of shape [K, D], where K is the code count
                and D is the code embedding dimension. Optionally [B, Lc, D] for
                a batched/sequence context.

        Returns:
            weighted_clusters (torch.Tensor):
                Attention-weighted code representations.
                Shape: [B, Lq, D] (or [Lq, D] if B collapses to 1).

        Notes:
            - If x are raw indices, ensure they are embedded to vectors compatible
              with the decoder dimension before cross-attention (or supply a
              context that already provides the corresponding code vectors).
            - When batch sizes differ between x and context, the smaller batch is
              internally repeated to match the larger one.
        """


        # Ensure both tensors have the same batch size
        if x.shape[0] != context.shape[0]:
            if x.shape[0] < context.shape[0]:
                # Repeat x to match the size of context
                repeat_count = (context.shape[0] // x.shape[0])+1  # Ceiling division
                x = x.repeat(repeat_count, 1)[:context.shape[0], ...]  # Repeat and trim
            else:
                # Repeat context to match the size of x
                repeat_count = (x.shape[0] // context.shape[0])+1  # Ceiling division
                context = context.repeat(repeat_count, 1)[:x.shape[0], ...]  # Repeat and trim

        x_in, context_in = x.unsqueeze(0), context.unsqueeze(0)
        # Apply VQAttention
        weighted_clusters = self.forward(x_in, context_in)
        
        return weighted_clusters.squeeze(0)  # Remove the batch dimension

