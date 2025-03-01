import math
import torch
import torch.nn as nn


class FluxMicroModel(nn.Module):
    def __init__(
        self,
        image_size=32,
        channels=3,
        embed_dim=128,
        num_layers=8,
        num_heads=4,
        ff_dim=512,
    ):
        """
        A simplified diffusion Transformer model for 32x32 images (FLUX Micro).
        :param image_size: Height/width of the image (32 for CIFAR-10).
        :param channels: Number of image channels (3 for RGB).
        :param embed_dim: Dimension of token embeddings.
        :param num_layers: Number of Transformer encoder layers.
        :param num_heads: Number of attention heads in each layer.
        :param ff_dim: Dimension of feed-forward network hidden layer in each Transformer block.
        """
        super().__init__()
        self.image_size = image_size
        self.channels = channels
        self.num_tokens = image_size * image_size  # 1024 for 32x32
        self.embed_dim = embed_dim

        # 1. Input token embedding: project pixel RGB (channels) to embed_dim
        self.token_proj = nn.Linear(channels, embed_dim)

        # 2. Positional embeddings for 1024 positions (learnable)
        self.pos_emb = nn.Parameter(torch.randn(1, self.num_tokens, embed_dim))

        # 3. Time-step embedding: project scalar t to embed_dim via sinusoidal encoding
        # We create a sinusoidal position embedding for time steps and then a linear layer.
        self.time_dim = embed_dim
        self.time_embed_linear = nn.Linear(
            embed_dim, embed_dim
        )  # learns to project time embedding to proper size

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 4. Output projection: project embed_dim back to pixel values (RGB channels)
        self.output_proj = nn.Linear(embed_dim, channels)

    def forward(self, x_noisy, t):
        """
        Forward pass of the diffusion model.
        :param x_noisy: Noisy input images, tensor of shape (batch, channels, 32, 32)
        :param t: Diffusion time-step for each image in the batch (tensor of shape (batch,))
        :return: Predicted noise for each image (same shape as x_noisy)
        """
        batch_size = x_noisy.shape[0]
        # Flatten the image to a sequence of tokens
        # x_noisy: (B, C, H, W) -> (B, H*W, C)
        x = x_noisy.view(batch_size, self.num_tokens, self.channels)
        # Project tokens to embedding dimension
        x = self.token_proj(x)  # shape: (B, num_tokens, embed_dim)

        # Add positional embedding (broadcast across batch)
        x = (
            x + self.pos_emb
        )  # pos_emb shape: (1, num_tokens, embed_dim) adds to (B, num_tokens, embed_dim)

        # Compute time-step embeddings
        # Create sinusoidal positional embedding for t
        # t is (batch,), we produce an embedding of size embed_dim for each t
        # Normalize t to [0,1] range for embedding (optional but helps numeric stability)
        T_max = 1000  # assuming max diffusion steps, can be set accordingly
        t_norm = t.float() / T_max  # (batch,)
        # Sinusoidal embedding (half dim for sin, half for cos)
        half_dim = self.time_dim // 2
        # prepare frequencies: log scale between 1 and 10000
        freqs = torch.exp(
            -math.log(10000) * torch.arange(0, half_dim, dtype=torch.float32) / half_dim
        ).to(t.device)
        # outer product t and freqs
        sinusoid = torch.outer(t_norm, freqs)  # shape: (batch, half_dim)
        sin_embed = torch.sin(sinusoid)
        cos_embed = torch.cos(sinusoid)
        time_emb = torch.cat(
            [sin_embed, cos_embed], dim=1
        )  # shape: (batch, embed_dim) assuming embed_dim is even
        if time_emb.shape[1] < self.time_dim:
            # if embed_dim is odd and we have one extra, pad with zero
            time_emb = nn.functional.pad(time_emb, (0, 1))
        # Project time embedding to embed_dim
        time_emb = self.time_embed_linear(time_emb)  # shape: (batch, embed_dim)

        # Add time embedding to each token embedding
        # Expand time_emb to (batch, num_tokens, embed_dim) and add
        x = x + time_emb.unsqueeze(
            1
        )  # unsqueeze to (batch, 1, embed_dim), then broadcast along sequence

        # Pass through Transformer encoder
        # The transformer will use self-attention and feed-forward layers
        x = self.transformer(x)  # shape: (batch, num_tokens, embed_dim)

        # Project back to pixel values
        x = self.output_proj(x)  # shape: (batch, num_tokens, channels)
        # Reshape back to image format
        x = x.view(
            batch_size, self.channels, self.image_size, self.image_size
        )  # (B, C, 32, 32)
        return x  # this is the predicted noise
