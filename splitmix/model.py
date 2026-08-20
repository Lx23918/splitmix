import math

import torch
import torch.nn as nn
from einops.layers.torch import Rearrange


class MeanCoreMixer(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.core_mlp = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        core = self.core_mlp(x.mean(dim=1))
        core = core.unsqueeze(1).expand(-1, x.size(1), -1)
        delta = self.fuse_mlp(torch.cat([x, core], dim=-1))
        return self.norm(x + delta)


class SoftmaxCoreMixer(nn.Module):
    def __init__(self, dim: int, core_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        core_dim = dim if core_dim is None else core_dim
        self.core_proj = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, core_dim),
        )
        self.fuse_mlp = nn.Sequential(
            nn.Linear(dim + core_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected = self.core_proj(x)
        weights = torch.softmax(projected, dim=1)
        core = torch.sum(projected * weights, dim=1)
        core = core.unsqueeze(1).expand(-1, x.size(1), -1)
        delta = self.fuse_mlp(torch.cat([x, core], dim=-1))
        return self.norm(x + delta)


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim + 1, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term[: dim // 2 + 1])
        pe[:, 1::2] = torch.cos(position * div_term[: dim // 2])
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = self.pe[: x.size(0), :].unsqueeze(1).repeat(1, x.size(1), 1).to(x.device)
        return x + pe


class EEGSelfAttention(nn.Module):
    def __init__(self, num_channels: int, sequence_length: int) -> None:
        super().__init__()
        self.position = PositionalEncoding(num_channels)
        layer = nn.TransformerEncoderLayer(d_model=num_channels, nhead=1)
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)
        self.sequence_length = sequence_length

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(2, 0, 1)
        x = self.position(x)
        x = self.encoder(x)
        return x.permute(1, 2, 0)


class ResidualAdd(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fn(x)


class PatchEmbedding(nn.Module):
    def __init__(self, num_channels: int, token_dim: int) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 40, (1, 25), (1, 1)),
            nn.AvgPool2d((1, 51), (1, 5)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Conv2d(40, 40, (num_channels, 1), (1, 1)),
            nn.BatchNorm2d(40),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Conv2d(40, token_dim, (1, 1), stride=(1, 1)),
            Rearrange("b e h w -> b (h w) e"),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x.unsqueeze(1))


class FlattenHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.contiguous().view(x.size(0), -1)


class ProjectionHead(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            ResidualAdd(
                nn.Sequential(
                    nn.GELU(),
                    nn.Linear(output_dim, output_dim),
                    nn.Dropout(dropout),
                )
            ),
            nn.LayerNorm(output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class SplitMixBackbone(nn.Module):
    def __init__(
        self,
        num_channels: int = 63,
        sequence_length: int = 250,
        token_dim: int = 40,
        use_cotar: bool = True,
        cotar_type: str = "mean",
    ) -> None:
        super().__init__()
        self.attention = EEGSelfAttention(num_channels=num_channels, sequence_length=sequence_length)
        self.use_cotar = use_cotar
        if use_cotar:
            if cotar_type == "mean":
                self.channel_mixer = MeanCoreMixer(sequence_length)
            elif cotar_type == "softmax":
                self.channel_mixer = SoftmaxCoreMixer(sequence_length)
            else:
                raise ValueError(f"Unknown cotar_type: {cotar_type}")
        self.patch_embedding = PatchEmbedding(num_channels=num_channels, token_dim=token_dim)
        self.flatten = FlattenHead()
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, sequence_length)
            self.output_dim = self.forward(dummy).shape[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attention(x)
        if self.use_cotar:
            x = self.channel_mixer(x)
        x = self.patch_embedding(x)
        return self.flatten(x)
