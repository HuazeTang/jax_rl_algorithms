import jax
import jax.numpy as jnp
from dataclasses import dataclass

import distrax
import flax.linen as nn

from utils.transition import Transition

@dataclass
class LossConfig:
    clip_eps: float
    vf_coef: float
    ent_coef: float

class PPOLoss:
    def __init__(self, network: nn.Module, config: LossConfig):
        self.network = network
        self.config = config
    
    def compute_loss(self, params: dict, batch: Transition, gae: jax.Array, targets: jax.Array) -> tuple:
        # Forward pass
        pi: distrax.MultivariateNormalDiag
        value: jax.Array
        pi, value = self.network.apply(params, batch.obs)
        log_prob = pi.log_prob(batch.action)

        # Value loss
        value_pred_clipped = batch.value + (value - batch.value).clip(-self.config.clip_eps, self.config.clip_eps)
        value_loss = 0.5 * jnp.maximum(
            jnp.square(value - targets),
            jnp.square(value_pred_clipped - targets)
        ).mean()

        # Actor loss
        ratio = jnp.exp(log_prob - batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor = -jnp.minimum(
            ratio * gae,
            jnp.clip(ratio, 1-self.config.clip_eps, 1+self.config.clip_eps) * gae
        ).mean()

        # Entropy
        entropy = pi.entropy().mean()

        # Total loss
        total_loss = loss_actor + self.config.vf_coef * value_loss - self.config.ent_coef * entropy
        return total_loss, (value_loss, loss_actor, entropy)
