import jax
import jax.numpy as jnp
import distrax
import numpy as np
import flax.linen as nn
from flax.linen.initializers import constant, orthogonal
from flax.core.frozen_dict import FrozenDict
from typing import Sequence, Tuple, Union, Any, Dict

from utils.mixed_precision import mixed_precision

class Actor(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))

        return pi 


class Critic(nn.Module):
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return jnp.squeeze(critic, axis=-1)


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    # @mixed_precision()
    @nn.compact
    def __call__(self, x):
        pi = Actor(action_dim=self.action_dim, activation=self.activation, name="actor")(x)
        critic = Critic(activation=self.activation, name="critic")(x)

        return pi, critic

def init_network(
        activation_type: str, rng: jax.Array, action_dim: int, obs_shape: Tuple
    ) -> Tuple[nn.Module, Union[FrozenDict, Dict[str, Any]]]:
    network = ActorCritic(
       action_dim=action_dim, activation=activation_type
    )
    init_x = jnp.zeros(obs_shape)
    network_params = network.init(rng, init_x)

    return network, network_params
