import numpy as np
import torch
import torch.nn as nn

from blocks import CrossAttention, Block
from pos_embed import get_1d_sincos_pos_embed_from_grid

class PerceiverCompressor(nn.Module):
    def __init__(self, token_dim, latent_dim, num_latents, num_cross_layers, num_latent_transformer_layers, num_heads=8, dropout=0.0, norm_layer=nn.LayerNorm):
        super(PerceiverCompressor, self).__init__()
        self.token_dim = token_dim
        self.latent_dim = latent_dim
        self.num_latents = num_latents
        self.num_heads = num_heads
        self.num_cross_layers = num_cross_layers
        self.num_latent_transformer_layers = num_latent_transformer_layers

        # Learnable latents
        self.latents = nn.Parameter(torch.randn(num_latents, latent_dim))

        # Cross-attention and latent transformer layers
        self.cross_attention_layers = nn.ModuleList()
        for _ in range(num_cross_layers):
            cross_attn_layer = nn.ModuleDict({
                'cross_attn': CrossAttention(
                    dim=latent_dim, num_heads=num_heads, qkv_bias=True, attn_drop=dropout, proj_drop=dropout
                ),
                'latent_transformer': nn.ModuleList([
                    Block(
                        dim=latent_dim, num_heads=num_heads, mlp_ratio=4.0, qkv_bias=True, drop=dropout, attn_drop=dropout, norm_layer=norm_layer
                    ) for _ in range(num_latent_transformer_layers)
                ]),
                'norm1': norm_layer(latent_dim),
                'norm2': norm_layer(latent_dim),
                'norm_x': norm_layer(latent_dim)
            })
            self.cross_attention_layers.append(cross_attn_layer)

    def forward(self, x, pos, image_ids):
        """
        Args:
            x (torch.Tensor): Input tensor of shape [B, P, C] where
                              B - batch size
                              P - total number of patches from all images
                              C - dimension of each visual token
            pos (torch.Tensor): Positional tensor of shape [B, P, 2] indicating positions
            image_ids (torch.Tensor): Tensor of shape [B, P] specifying which image each patch belongs to
        Returns:
            torch.Tensor: Compressed latent representation of shape [B, L, D] where
                          L - number of latents
                          D - dimension of each latent representation
        """
        B, P, C = x.shape

        # Repeat the latents for each batch
        latents = self.latents.unsqueeze(0).expand(B, -1, -1)

        # Compute image positional encoding dynamically
        num_images = (torch.max(image_ids) + 1).cpu().item()
        image_pos_emb = torch.from_numpy(
            get_1d_sincos_pos_embed_from_grid(self.token_dim, np.arange(num_images))
        ).float().to(x.device)

        # Add image positional encoding to distinguish image sources
        image_pos = image_pos_emb[image_ids]
        x += image_pos

        # Alternate between cross-attention and latent transformer layers
        for layer in self.cross_attention_layers:
            # first, compress the image tokens into latents
            latents = layer['cross_attn'](query=layer['norm1'](latents), key=layer['norm_x'](x), value=layer['norm_x'](x), qpos=None, kpos=pos)
            # then, self-attend the latents to refine them
            for latent_transformer_layer in layer['latent_transformer']:
                latents = latent_transformer_layer(x=layer['norm2'](latents), xpos=None)

        return latents





# Example usage
B, P, C = 2, 100*256, 768  # Example dimensions (batch size, total patches, token dimension)
L, D = 1000, 768  # Latent dimensions
num_cross_layers = 4
num_latent_transformer_layers = 2
num_heads = 8
dropout = 0.1

compressor = PerceiverCompressor(token_dim=C, latent_dim=D, num_latents=L, num_cross_layers=num_cross_layers, num_latent_transformer_layers=num_latent_transformer_layers, num_heads=num_heads, dropout=dropout).cuda()
input_tensor = torch.randn(B, P, C).cuda()
pos_tensor = torch.randn(B, P, 2).cuda()  # Example positional tensor
image_ids = torch.tensor([[i] * 256 for i in range(100)] * B).cuda().reshape(B, -1)  # Example image IDs for patches
output_tensor = compressor(input_tensor, pos_tensor, image_ids)
print(output_tensor.shape)  # Should print torch.Size([1, 1000, 768])
