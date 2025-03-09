import jax.numpy as jnp
from typing import NamedTuple

class Transition(NamedTuple):
    """
    A named tuple representing a transition in a reinforcement learning environment.
    """
    done: jnp.ndarray     # A boolean array indicating whether the episode has ended.
    action: jnp.ndarray   # The action taken by the agent in the current state.
    value: jnp.ndarray    # The predicted value of the current state by the value network.
    reward: jnp.ndarray   #  The reward received after taking the action.
    log_prob: jnp.ndarray # The log probability of the action taken by the policy network.
    obs: jnp.ndarray      # The observation of the current state.
    info: jnp.ndarray     # Additional information or metadata about the transition.
