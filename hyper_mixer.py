from typing import Type, Tuple, Optional
from functools import partial

import torch
import torch.nn as nn
import timm


class HyperMixerBlock(nn.Module):
    """
    This class implements a HyperMixer block.
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: Tuple[float, float] = (0.5, 4.0),
            norm_layer: Type = partial(nn.LayerNorm, eps=1e-06),
            act_layer: Type = nn.GELU,
            drop: float = 0.,
            drop_path: float = 0.
    ) -> None:
        """
        Constructor method
        :param dim (int): Channel dimension
        :param mlp_ratio (Tuple[int, int]): Ratio of hidden dim. of the hyper mixer layer and MLP. Default = (0.5, 4.0)
        :param norm_layer (Type): Type of normalization to be used. Default = nn.LayerNorm
        :param act_layer (Type): Type of activation layer to be used. Default = nn.GELU
        :param drop (float): Dropout rate. Default = 0.
        :param drop_path (float): Dropout path rate. Default = 0.
        """
        # Call super constructor
        super(HyperMixerBlock, self).__init__()
        # Init layers
        tokens_dim, channels_dim = [int(x * dim) for x in timm.models.layers.to_2tuple(mlp_ratio)]
        self.norm1: nn.Module = norm_layer(dim)
        self.mlp_tokens: nn.Module = HyperMixer(dim=dim, hidden_dim=tokens_dim, act_layer=act_layer, drop=drop)
        self.drop_path = timm.models.layers.DropPath(drop_prob=drop_path)
        self.norm2: nn.Module = norm_layer(dim)
        self.mlp_channels: nn.Module = timm.models.layers.Mlp(in_features=dim, hidden_features=channels_dim,
                                                              act_layer=act_layer, drop=drop)

    def forward(
            self,
            x: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :return (torch.Tensor): Output tensor of the shape [batch size, tokens, channels]
        """
        x: torch.Tensor = self.norm1(x)
        x: torch.Tensor = x + self.drop_path(self.mlp_tokens(x))
        x: torch.Tensor = x + self.drop_path(self.mlp_channels(self.norm2(x)))
        return x


class HyperMixer(nn.Module):
    """
    This class implements the Hyper Mixer layer.
    """

    def __init__(
            self,
            dim: int,
            hidden_dim: int,
            act_layer: Type = nn.GELU,
            drop: float = 0.,
    ) -> None:
        """
        Constructor method
        :param dim (int): Channel dimension
        :param hidden_dim (int): Size of hidden dimension
        :param act_layer (Type): Type of activation function to be used
        :param drop (float): Dropout rate
        """
        # Call super constructor
        super(HyperMixer, self).__init__()
        # Init modules
        self.mlp_1: nn.Module = timm.models.layers.Mlp(in_features=dim, out_features=hidden_dim, act_layer=nn.GELU,
                                                       drop=drop)
        self.drop = nn.Dropout(drop)
        self.act = act_layer()

    def forward(
            self,
            x: torch.Tensor,
            pos_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        :param x (torch.Tensor): Input tensor of the shape [batch size, tokens, channels]
        :param pos_emb (torch.Tensor): Optional positional embeddings for y
        :return (torch.Tensor): Output tensor of the shape
        """
        # Compute weights
        weights: torch.Tensor = self.mlp_1(x + pos_emb if pos_emb else x)
        # Map input with weights and activate
        x: torch.Tensor = self.drop(self.act(weights.transpose(1, 2) @ x))
        x: torch.Tensor = self.drop(weights @ x)
        return x


if __name__ == '__main__':
    hyper_mixer_block = HyperMixerBlock(dim=32)
    input = torch.rand(3, 256, 32)
    output = hyper_mixer_block(input)
    print(output.shape)
