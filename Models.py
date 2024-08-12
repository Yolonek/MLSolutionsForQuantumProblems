import jax
import jax.numpy as jnp
import flax.linen as nn
import netket.nn as nknn
import netket as nk


class RestrictedBoltzmannMachine(nn.Module):
    param_dtype = jnp.float64
    alpha = 1
    stddev = 0.01

    @nn.compact
    def __call__(self, x):
        layer = nn.Dense(name='Dense',
                         features=int(self.alpha * x.shape[-1]),
                         param_dtype=self.param_dtype,
                         use_bias=True,
                         kernel_init=jax.nn.initializers.normal(stddev=self.stddev),
                         bias_init=jax.nn.initializers.normal(stddev=self.stddev))
        bias = self.param('visible_bias',
                          self.jax.nn.initializers.normal(stddev=self.stddev),
                          (x.shape[-1],),
                          self.param_dtype)
        return jnp.sum(nknn.log_cosh(layer(x)), axis=-1) + jnp.dot(x, bias)


class FeedForwardNeuralNetwork(nn.Module):
    param_dtype = jnp.float64
    layer_alpha = [1]
    stddev = 0.01

    @nn.compact
    def __call__(self, x):
        activation = nknn.log_cosh
        for index, alpha in enumerate(self.layer_alpha):
            x = activation(nn.Dense(f'Dense{index}',
                                    features=int(alpha * x.shape[-1]),
                                    param_dtype=self.param_dtype,
                                    use_bias=True,
                                    kernel_init=jax.nn.initializers.normal(stddev=self.stddev),
                                    bias_init=jax.nn.initializers.normal(stddev=self.stddev))(x))
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
        batch_size, seq_len, emb_dim = x.shape
        qkv = nn.Dense(self.num_heads * self.head_dim * 3)(x)
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3 * self.head_dim)
        query, key, value = jnp.split(qkv, 3, axis=-1)

        attention_outputs, _ = DotProductAttention()(query, key, value)
        attention_outputs = attention_outputs.reshape(batch_size, seq_len, -1)
        return nn.Dense(self.head_dim * self.num_heads)(attention_outputs)


class TransformerBlock(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int

    @nn.compact
    def __call__(self, x, deterministic=True):
        y = nn.LayerNorm()(x)
        y = MultiHeadAttention(num_heads=self.num_heads, head_dim=self.head_dim)(y)
        x = x + y

        y = nn.LayerNorm()(x)
        y = nn.Dense(self.mlp_dim)(y)
        y = nn.relu(y)
        y = nn.Dense(self.head_dim * self.num_heads)(y)
        return x + y


class Psiformer(nn.Module):
    num_heads: int
    head_dim: int
    mlp_dim: int
    num_layers: int
    max_length: int

    @nn.compact
    def __call__(self, x):
        pos_embedding = self.param('pos_embedding',
                                   nn.initializers.normal(stddev=0.02),
                                   (1, x.shape[1], self.num_heads * self.head_dim))
        x = nn.Embed(self.max_length, self.num_heads * self.head_dim)(x) + pos_embedding

        for _ in range(self.num_layers):
            x = TransformerBlock(self.num_heads, self.head_dim, self.mlp_dim)(x)

        x = nn.LayerNorm()(x)
        return x.mean(axis=1)
