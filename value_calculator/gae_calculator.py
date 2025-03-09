import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Tuple

from utils.transition import Transition

@dataclass
class GAEConfig:
    gamma: float
    gae_lambda: float

class GAECalculator:
    def __init__(self, config: GAEConfig):
        self.config = config

    def calculate_single(
            self, 
            gae_and_next_value: jax.Array, 
            transition: Transition
        ) -> Tuple[Tuple[jax.Array, jax.Array], jax.Array]:
        gae, next_value = gae_and_next_value
        done, value, reward = transition.done, transition.value, transition.reward
        gamma, gae_lambda = self.config.gamma, self.config.gae_lambda
        delta = (
            reward
            + gamma * next_value * (1 - done)
            - value
        )
        gae = (
            delta 
            + gamma * gae_lambda * (1 - done) * gae
        )
        return (gae, value), gae

    def calculate_batch(
            self, 
            traj_batch: Transition, 
            last_val: jax.Array
        ) -> jax.Array:
        _, advantages = jax.lax.scan(
            lambda carry, trans: self.calculate_single(carry, trans),
            (jnp.zeros_like(last_val), last_val),
            traj_batch,
            reverse=True,
            unroll=16
        )
        return advantages
