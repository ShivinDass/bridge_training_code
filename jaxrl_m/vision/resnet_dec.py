import functools as ft
from functools import partial
from typing import Any, Callable, Sequence, Tuple

import jax
import flax.linen as nn
import jax.numpy as jnp
import numpy as np

from jaxrl_m.vision.film_conditioning_layer import FilmConditioning
from jaxrl_m.common.common import default_init

ModuleDef = Any


class MyGroupNorm(nn.GroupNorm):
    def __call__(self, x):
        if x.ndim == 3:
            x = x[jnp.newaxis]
            x = super().__call__(x)
            return x[0]
        else:
            return super().__call__(x)


def ResizedConv2d(x, conv, filters, scale_factor):
    # x.shape = (N, H, W, C)
    x = jax.image.resize(x, (x.shape[0], x.shape[1] * scale_factor, x.shape[2] * scale_factor, x.shape[3]), method="nearest", antialias=False)
    x = conv(filters, (3, 3), strides=1, padding=1, use_bias=True)(x)
    return x


# A Jax reimplementation of https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py
class ResNetDecBlock(nn.Module):

    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), strides=1, padding=1)(x)
        y = self.norm()(y)
        y = self.act(y)

        if self.strides[0] == 1:
            y = self.conv(self.filters, (3, 3), strides=1, padding=1)(y)
            y = self.norm()(y)
        else:
            y = ResizedConv2d(y, self.conv, int(self.filters / self.strides[0]), self.strides[0])
            y = self.norm()(y)
            residual = ResizedConv2d(residual, self.conv, int(self.filters / self.strides[0]), self.strides[0])
            residual = self.norm()(residual)
        return self.act(residual + y)


class ResNetDecoder(nn.Module):

    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_output_channels: int
    output_hw: int
    only_pos_output: bool = False
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: str = "relu"
    conv: ModuleDef = nn.Conv
    # NOTE: we were using batch norm
    norm: str = "group" 

    @nn.compact
    def __call__(self, embeddings: jnp.ndarray, train: bool = True, cond_var=None):

        batch_size = embeddings.shape[0]
        x = nn.Dense(512, kernel_init=default_init())(embeddings.reshape(batch_size, -1))
        x = x.reshape(batch_size, 1, 1, 512)
        x = jax.image.resize(x, (batch_size, 5, 5, 512), method="nearest", antialias=False)

        conv = partial(
            self.conv,
            use_bias=False,
            dtype=self.dtype,
            kernel_init=nn.initializers.kaiming_normal(),
        )
        if self.norm == "batch":
            raise NotImplementedError
        elif self.norm == "group":
            norm = partial(MyGroupNorm, num_groups=4, epsilon=1e-5, dtype=self.dtype)
        elif self.norm == "layer":
            norm = partial(nn.LayerNorm, epsilon=1e-5, dtype=self.dtype)
        else:
            raise ValueError("norm not found")
        act = getattr(nn, self.act)

        for i in range(len(self.stage_sizes)-1, -1, -1):
            block_size = self.stage_sizes[i]
            for j in range(block_size):
                stride = (2, 2) if i > 0 and j == block_size-1 else (1, 1)
                x = self.block_cls(
                    self.num_filters * 2 ** i,
                    strides=stride,
                    conv=conv,
                    norm=norm,
                    act=act,
                )(x)
        
        x = ResizedConv2d(x, conv, self.num_output_channels, scale_factor=2)
        if self.only_pos_output:
            x = jax.nn.sigmoid(x)
        x = jax.image.resize(x, (batch_size, self.output_hw, self.output_hw, self.num_output_channels), method="nearest", antialias=False) 

        return x


resnetdec_configs = {
    "resnet-18-dec": ft.partial(
        ResNetDecoder, stage_sizes=(2, 2, 2, 2), block_cls=ResNetDecBlock
    ),
    "resnet-34-dec": ft.partial(
        ResNetDecoder, stage_sizes=(3, 4, 6, 3), block_cls=ResNetDecBlock
    )
}
