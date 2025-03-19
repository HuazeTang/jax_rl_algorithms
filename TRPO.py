import jax
import jax.numpy as jnp
from jax import config as jax_config

import flax.linen as nn
from flax.training.train_state import TrainState

import distrax
import datetime
import functools
from network.actor_critic import init_network
from typing import Optional, Tuple, Callable, Dict, Any
from tensorboardX import SummaryWriter
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

from utils.config import AlgoConfig, ScheduleType, OptimizerType
from utils.config import make_schedule_from_config as make_schedule
from utils.config import make_tx_from_config as init_tx
from utils.transition import Transition

from value_calculator.gae_calculator import GAECalculator, GAEConfig
from loss.trpo_loss import TRPOLoss, LossConfig


# Setting up related: Env / Tx
def make_env(config: AlgoConfig) -> Tuple[VecEnv, Optional[Dict]]:
    env, env_params = BraxGymnaxWrapper(config.ENV_NAME), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config.NORMALIZE_ENV:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config.GAMMA)
    
    return env, env_params

# Interaction related: Env_step
def env_step(
        network: nn.Module,          # network of actor critic
        env: VecEnv,                 # envrionment
        env_params: Optional[Dict],  # envrionment params
        config: AlgoConfig,          # golbal configs
        rng: jax.Array,              # random numbers for stochastic env
        runner_state: Tuple,         # current state
        unsed: None,                 # place holder for jax.lax.scan
    ):
    # EXTRACT SURRENT STATE
    train_state, env_state, last_obs, rng = runner_state

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi, value = network.apply(train_state.params, last_obs)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, config.NUM_ENVS)
    obs, env_state, reward, done, info = env.step(
        rng_step, env_state, action, env_params
    )
    transition = Transition(
        done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obs, rng)
    return runner_state, transition

def make_env_step(
        network: nn.Module, env: VecEnv, env_params: Optional[Dict], config: AlgoConfig, rng: jax.Array
    ):
    return functools.partial(
        env_step, network, env, env_params, config, rng
    )

# Pipeline related: update network in minibatch / epoch / step
def update_minibatch(
    network: nn.Module,
    config: AlgoConfig,
    train_state: TrainState, 
    batch_info: Tuple[TrainState, jax.Array, jax.Array]
) -> Tuple[TrainState, jax.Array]:
    traj_batch, advantages, targets = batch_info

    trpo_loss = TRPOLoss(
        network=network,
        config=LossConfig(
            clip_eps=config.CLIP_EPS, vf_coef=config.VF_COEF, ent_coef=config.ENT_COEF
        )
    )

    total_loss, grads  = trpo_loss.update_step(
        params=train_state.params,
        batch=traj_batch,
        advantages=advantages,
        targets=targets,
    )

    train_state = train_state.apply_gradients(grads=grads)

    return train_state, total_loss

def make_update_minibatch(network: nn.Module, config: AlgoConfig):
    return functools.partial(
        update_minibatch, network, config
    )

def update_epoch(
    network: nn.Module,
    config: AlgoConfig,
    update_state: Tuple[TrainState, Transition, jax.Array, jax.Array, jax.Array],
    unsed: None,                 # place holder for jax.lax.scan
):
    train_state, traj_batch, advantages, targets, rng = update_state
    rng, _rng = jax.random.split(rng)
    batch_size = config.MINIBATCH_SIZE * config.NUM_MINIBATCHES
    assert (
        batch_size == config.NUM_STEPS * config.NUM_ENVS
    ), f"batch size must be equal to number of steps * number of envs. but batch size {batch_size} not match steps * number {config.NUM_STEPS} * {config.NUM_ENVS}"
    permutation = jax.random.permutation(_rng, batch_size)
    batch_data = (traj_batch, advantages, targets)
    batch_data = jax.tree_util.tree_map(
        lambda x: x.reshape((batch_size,) + x.shape[2:]), batch_data
    )
    shuffled_batch_data = jax.tree_util.tree_map(
        lambda x: jnp.take(x, permutation, axis=0), batch_data
    )
    minibatches = jax.tree_util.tree_map(
        lambda x: jnp.reshape(
            x, [config.NUM_MINIBATCHES, -1] + list(x.shape[1:])
        ),
        shuffled_batch_data,
    )
    minibatch_fn = make_update_minibatch(network=network, config=config)
    train_state, total_loss = jax.lax.scan(
        minibatch_fn, train_state, minibatches
    )
    update_state = (train_state, traj_batch, advantages, targets, rng)
    return update_state, total_loss

def make_update_epoch(network: nn.Module, config: AlgoConfig):
    return functools.partial(
        update_epoch, network, config
    )

def update_step(
    network: nn.Module, 
    env: VecEnv, 
    env_params: Optional[Dict], 
    config: AlgoConfig, 
    rng: jax.Array,
    callback_fn: Optional[Callable],
    runner_state: Tuple, 
    unused
):
    # COLLECT TRAJECTORIES
    env_step_fn = make_env_step(network=network, env=env, env_params=env_params, config=config, rng=rng)

    runner_state: Tuple
    traj_batch: Transition
    runner_state, traj_batch = jax.lax.scan(
        env_step_fn, runner_state, None, config.NUM_STEPS
    )

    # CALCULATE ADVANTAGE AND TARGET
    train_state: TrainState
    train_state, env_state, last_obs, rng = runner_state
    _, last_val = network.apply(train_state.params, last_obs)

    advantages = value_calculator.calculate_batch(traj_batch=traj_batch, last_val=last_val)
    targets = advantages + traj_batch.value

    # UPDATE NETWORK
    update_epoch_fn = make_update_epoch(network=network, config=config)
    
    update_state = (train_state, traj_batch, advantages, targets, rng)
    update_state, loss_info = jax.lax.scan(
        update_epoch_fn, update_state, None, config.UPDATE_EPOCHS
    )

    train_state = update_state[0]
    step = train_state.step
    metric = traj_batch.info
    rng = update_state[-1]

    if config.DEBUG:
        info_total = (metric, step, loss_info)
        jax.debug.callback(callback_fn, info_total)

    runner_state = (train_state, env_state, last_obs, rng)
    return runner_state, metric

def make_update_step(
    network: nn.Module, 
    env: VecEnv, 
    env_params: Optional[Dict], 
    config: AlgoConfig, 
    rng: jax.Array,
    callback_fn: Callable
) -> Callable:
    return functools.partial(
        update_step, network, env, env_params, config, rng, callback_fn
    )

def callback(
    writer: SummaryWriter, 
    config: AlgoConfig, 
    info_total: Tuple[Dict[str, Any], int, jax.Array],
):
    info, step_, loss_info = info_total
    return_values = info["returned_episode_returns"][
        info["returned_episode"]
    ]
    timesteps = (
        info["timestep"][info["returned_episode"]] * config.NUM_ENVS
    )

    if len(timesteps)>=1:
        if config.ENV_NAME=="humanoidstandup":
            for t in range(len(timesteps)):
                writer.add_scalar('episodic return', return_values[t], timesteps[t])
        else:
            writer.add_scalar('episodic return', return_values[0], timesteps[0])
    writer.add_scalar('Loss/total_loss', loss_info[0].mean(), step_)
    writer.add_scalar('Loss/value_loss', loss_info[0][0].mean(), step_)
    writer.add_scalar('Loss/actor_loss', loss_info[0][1].mean(), step_)
    writer.add_scalar('Loss/entropy', loss_info[0][2].mean(), step_)

def make_callback(writer: SummaryWriter, config: AlgoConfig):
    return functools.partial(
        callback, writer, config
    )

# Main train function
def make_train(config: AlgoConfig) -> Callable:
    # Init env
    env, env_params = make_env(config=config)

    # Init schdule
    schdule_fn = make_schedule(config=config)

    # Init optimizer
    optimizer = init_tx(config=config, schedule_fn=schdule_fn)

    def train(rng):
        # Init writer
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(logdir=f'./logs/TRPO_{config.ENV_NAME}_lr{config.LR}_steps{config.NUM_STEPS}_epochs{config.UPDATE_EPOCHS}_{timestamp}')  

        # Init network
        rng, _rng = jax.random.split(rng)
        network, network_params = init_network(
            activation_type=config.ACTIVATION, 
            rng=_rng, 
            action_dim=env.action_space(env_params).shape[0], 
            obs_shape=env.observation_space(env_params).shape
        )

        # Init train state by network params
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=optimizer,
        )

        # Init env
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.NUM_ENVS)
        obs, env_state = env.reset(reset_rng, env_params)

        # Train loop
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        callback_fn = make_callback(writer=writer, config=config)
        update_step_fn = make_update_step(
            network=network, env=env, env_params=env_params, config=config, rng=_rng, callback_fn=callback_fn
        )

        runner_state, metric = jax.lax.scan(
            update_step_fn, runner_state, None, config.NUM_UPDATES
        )

        writer.close()

        return {"runner_state": runner_state, "metrics": metric}

    return train

if __name__ == "__main__":
    algo_config = AlgoConfig(
        NUM_ENVS=2048, 
        NUM_STEPS=40, 
        TOTAL_TIMESTEPS=1e8, 
        UPDATE_EPOCHS=4, 
        NUM_MINIBATCHES=128, 
        GAMMA=0.99, 
        GAE_LAMBDA=0.95, 
        CLIP_EPS=1e5, 
        ENT_COEF=0.0, 
        VF_COEF=0.5, 
        LR=3e-4, 
        MAX_GRAD_NORM=0.5, 
        TX_TYPE=OptimizerType.NotUsing,
        ACTIVATION="tanh", 
        ANNEAL_LR=False, 
        SCHEDULE_TYPE=ScheduleType.Linear,
        ENV_NAME="walker2d", 
        NORMALIZE_ENV=True, 
        DEBUG=True, 
    )

    value_calculator = GAECalculator(
        GAEConfig(gamma=algo_config.GAMMA, gae_lambda=algo_config.GAE_LAMBDA)
    )

    # jax.config.update("jax_debug_nans", True)
    # jax.config.update("jax_debug_infs", True)

    rng = jax.random.PRNGKey(20)
    # train_jit = jax.jit(make_train(algo_config), device=jax.devices('gpu')[0])
    # out = train_jit(rng)
    train = make_train(algo_config)
    out = train(rng)
