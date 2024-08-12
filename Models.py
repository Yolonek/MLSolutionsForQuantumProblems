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

