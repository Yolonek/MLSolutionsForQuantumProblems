import jax
import jax.numpy as jnp
import flax.linen as nn
import netket.nn as nknn
import netket as nk
from typing import Union, Any, Tuple, Callable


class RestrictedBoltzmannMachine(nn.Module):
    param_dtype: Any = jnp.float64
    alpha: Union[float, int] = 1
    stddev: float = 0.01

    @nn.compact
    def __call__(self, x):
        layer = nn.Dense(name='Dense',
                         features=int(self.alpha * x.shape[-1]),
                         param_dtype=self.param_dtype,
                         use_bias=True,
                         kernel_init=jax.nn.initializers.normal(stddev=self.stddev),
                         bias_init=jax.nn.initializers.normal(stddev=self.stddev))
        bias = self.param('visible_bias',
                          jax.nn.initializers.normal(stddev=self.stddev),
                          (x.shape[-1],),
                          self.param_dtype)
        return jnp.sum(nknn.log_cosh(layer(x)), axis=-1) + jnp.dot(x, bias)


class FeedForwardNeuralNetwork(nn.Module):
    param_dtype: Any = jnp.float64
    layer_alpha: Tuple[int] = (1,)
    stddev: float = 0.01
    activation: Callable = nknn.log_cosh

    @nn.compact
    def __call__(self, x):
        for index, alpha in enumerate(self.layer_alpha):
            x = self.activation(nn.Dense(name=f'Dense{index + 1}',
                                         features=int(alpha * x.shape[-1]),
                                         param_dtype=self.param_dtype,
                                         use_bias=True,
                                         kernel_init=jax.nn.initializers.normal(stddev=self.stddev),
                                         bias_init=jax.nn.initializers.normal(stddev=self.stddev))(x))
        return jnp.sum(x, axis=-1)


class TransformerBlock(nn.Module):
    num_heads: int = 1
    head_dim: int = 1
    alpha: Union[float, int] = 1
    param_dtype: Any = jnp.float64
    activation: Callable = nknn.log_cosh
    stddev: float = 0.01

    @nn.compact
    def __call__(self, x):
        residual = x
        dense_params = dict(param_dtype=self.param_dtype,
                            kernel_init=jax.nn.initializers.normal(stddev=self.stddev),
                            bias_init=jax.nn.initializers.normal(stddev=self.stddev))
        x = nn.LayerNorm()(x)
        num_features = int(self.alpha * x.shape[-1])
        qkv_features = self.num_heads * self.head_dim * 3 * x.shape[-1]
        qkv = nn.Dense(features=qkv_features,
                       use_bias=False,
                       **dense_params)(x)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        x = nn.MultiHeadDotProductAttention(num_heads=self.num_heads,
                                            # qkv_features=qkv_features,
                                            **dense_params)(q, k, v)
        x = nn.Dense(features=num_features,
                     use_bias=True,
                     **dense_params)(x)
        x = self.activation(x)
        return nn.Dense(features=residual.shape[-1], use_bias=False, **dense_params)(x) + residual


class Psiformer(nn.Module):
    transformer_blocks: int = 1
    num_heads: int = 2
    alpha: Union[float, int] = 1
    param_dtype: Any = jnp.float64
    activation: Callable = nknn.log_cosh
    stddev: float = 0.01

    @nn.compact
    def __call__(self, x):
        print(x.shape)
        for _ in range(self.transformer_blocks):
            x = TransformerBlock(
                num_heads=self.num_heads,
                alpha=self.alpha,
                param_dtype=self.param_dtype,
                activation=self.activation,
                stddev=self.stddev
            )(x)
        return jnp.sum(x, axis=-1)


class DotProductAttention(nn.Module):
    @nn.compact
    def __call__(self, query, key, value):
        d_k = query.shape[-1]
        scores = jnp.matmul(query, key.swapaxes(-2, -1)) / jnp.sqrt(d_k)
        attention_weights = nn.softmax(scores, axis=-1)
        return jnp.matmul(attention_weights, value), attention_weights

class MultiHeadAttention(nn.Module):
    num_heads: int
    head_dim: int

    @nn.compact
    def __call__(self, x):
        qkv = nn.Dense(self.head_dim * self.num_heads * 3)(x)
        query, key, value = jnp.split(qkv, 3, axis=-1)
        attention_outputs, _ = DotProductAttention()(query, key, value)
        # print(attention_outputs.shape)
        # attention_outputs = attention_outputs.reshape(1,-1)
        # print(attention_outputs.shape)
        return nn.Dense(features=self.head_dim * self.num_heads)(attention_outputs)


class TransformerLayer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x):
        y = nn.LayerNorm()(x)
        y = MultiHeadAttention(num_heads=self.num_heads, head_dim=self.head_dim)(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nknn.log_cosh(y)
        # y = nn.Dense(self.head_dim + self.num_heads)(y)
        return x + y


class PsiformerExtended(nn.Module):
    num_heads: int = 4
    head_dim: int = 16
    mlp_dim: int = 64
    num_layers: int = 1

    @nn.compact
    def __call__(self, x):
        embedding_dim = self.num_heads * self.head_dim
        pos_embedding = self.param('pos_embedding',
                                   nn.initializers.normal(stddev=0.02),
                                   (1, embedding_dim))
        x = nn.Dense(embedding_dim)(x)
        x = x + pos_embedding
        for _ in range(self.num_layers):
            x = TransformerLayer(self.num_heads, self.head_dim, self.mlp_dim)(x)

        x = nn.LayerNorm()(x)
        return x.mean(axis=1)
