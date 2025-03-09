import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
import datetime
import distrax
from jax import vmap
from tensorboardX import SummaryWriter
from wrappers import (
    LogWrapper,
    BraxGymnaxWrapper,
    VecEnv,
    NormalizeVecObservation,
    NormalizeVecReward,
    ClipAction,
)

# Gaussian Kernel Function
def gaussian_kernel(s, x, RBF_sigma):
    """Compute Gaussian kernel K(s, x)"""
    diff = s - x
    return jnp.exp(-0.5 *jnp.sum(diff * diff * RBF_sigma))

# Summation of Gaussian Kernels
def sum_gaussian_kernels(alpha, s_set, x, RBF_sigma):
    """Compute the summation of Gaussian kernels \Sigma \{alpha_k} * K(s_k, x)"""
    kernel_values = vmap(lambda alpha_k, s_k: alpha_k * gaussian_kernel(s_k, x, RBF_sigma))(alpha, s_set)
    return jnp.sum(kernel_values, axis=0)



class HFunction(nn.Module):
    RBF_sigma: jnp.ndarray
    action_dim: Sequence[int]
    num_updates: int
    feature_dim: int
    @nn.compact
    def __call__(self, x):
        alpha = self.param('alpha', nn.initializers.zeros, (self.num_updates, self.action_dim))
        s_set = self.param('s_set', nn.initializers.zeros, (self.num_updates, self.feature_dim))

        if x.ndim == 1:  
            return sum_gaussian_kernels(alpha, s_set, x, self.RBF_sigma)
        else: 
            return vmap(lambda xi: sum_gaussian_kernels(alpha, s_set, xi, self.RBF_sigma))(x)

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    feature_dim: int
    RBF_sigma: jnp.ndarray
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
            self.feature_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        h_layer = HFunction(RBF_sigma=self.RBF_sigma, action_dim=self.action_dim, num_updates=config["NUM_UPDATES"], feature_dim=self.feature_dim)
        h_output = h_layer(actor_mean) 
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(h_output, jnp.exp(actor_logtstd))

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

        return pi, jnp.squeeze(critic, axis=-1), h_output, actor_mean


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray



def make_train(config):
    config["NUM_UPDATES"] = (
        int(config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    ))
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["RKHS_UPDATE"] = int(config["NUM_UPDATES"]*config["UPDATE_PERCENT"])
    initial_lr = config["INITIAL_LR"]
    env, env_params = BraxGymnaxWrapper(config["ENV_NAME"]), None
    env = LogWrapper(env)
    env = ClipAction(env)
    env = VecEnv(env)
    if config["NORMALIZE_ENV"]:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, config["GAMMA"])
    RBF_sigma = jnp.ones(config["FEATURE_DIM"])* config["RBF_RATE"]

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(logdir=f'./logs/{config["ENV_NAME"]}_lr{config["INITIAL_LR"]}_UPDATE_PERCENT{config["UPDATE_PERCENT"]}_CLIP_EPS{config["CLIP_EPS"]}_CLIP_EPS_RATIO{config["CLIP_EPS_RATIO"]}_MAX_GRAD_NORM{config["MAX_GRAD_NORM"]}_{timestamp}')  # Specify your log directory
        network = ActorCritic(
            env.action_space(env_params).shape[0], feature_dim=config["FEATURE_DIM"], RBF_sigma=RBF_sigma, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros(env.observation_space(env_params).shape)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            def create_param_labels(params):
                param_labels = {}
                for key, val in params.items():
                    if isinstance(val, dict):
                        param_labels[key] = create_param_labels(val)
                    else:
                        label = 'frozen' if 'alpha' in key or 's_set' in key else 'trainable'
                        param_labels[key] = label
                return param_labels
            param_labels = create_param_labels(network_params)
            tx = optax.multi_transform(
            {
                'trainable': optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            ),  
                'frozen': optax.set_to_zero(),                    
            },
            param_labels=param_labels
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)


        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, rng, current_index = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value, _, _ = network.apply(train_state.params, last_obs)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv, env_state, reward, done, info = env.step(
                    rng_step, env_state, action, env_params
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_obs, info
                )
                runner_state = (train_state, env_state, obsv, rng, current_index)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, rng, current_index = runner_state
            _, last_val, _, _ = network.apply(train_state.params, last_obs)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        # RERUN NETWORK
                        pi, value, _, _ = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * value_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss
            
            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )

            train_state = update_state[0]
            # 更新 alpha 和 s_set 参数
            def update_h_params(params, action, obs, GAE, sigma, h_output, actor_mean, ratio):
                """Update the alpha and s_set parameters."""
                Diff = action - h_output
                lr = initial_lr * (1 - current_index / int(config["RKHS_UPDATE"]))
                lr = jax.nn.relu(lr)

                ratio = jnp.clip(ratio, 0, 1.0 + config["CLIP_EPS_RATIO"])
                new_alpha = ratio * GAE * Diff / sigma  
                new_s_set = actor_mean
                new_alpha = lr * new_alpha
                params['HFunction_0']['alpha'] = jax.lax.dynamic_update_slice(params['HFunction_0']['alpha'], jnp.expand_dims(new_alpha,0), (current_index, 0))
                params['HFunction_0']['s_set'] = jax.lax.dynamic_update_slice(params['HFunction_0']['s_set'], jnp.expand_dims(new_s_set,0), (current_index, 0))
                
                return params
            gae_mean = jnp.mean(advantages, axis=1, keepdims=True)
            gae_std = jnp.std(advantages, axis=1, keepdims=True)

            gae_normalized = (advantages - gae_mean) / (gae_std + 1e-8)
            GAE = gae_normalized[0,0]
            selected_obs = traj_batch.obs[0,0,:]
            selected_action = traj_batch.action[0,0,:]
            pi, G, h_output, actor_mean = network.apply(train_state.params, selected_obs)
            log_prob = pi.log_prob(selected_action)
            ratio = jnp.exp(log_prob - traj_batch.log_prob[0,0])
            sigma = jnp.exp(train_state.params['params']['log_std'])
            train_state.params['params'] = update_h_params(train_state.params['params'], selected_action, selected_obs, GAE, sigma, h_output, actor_mean, ratio)
            current_index +=1

            
            step = train_state.step
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):
                info_total = (metric, step, loss_info)
                def callback(info_total):
                    info, step_, loss_info = info_total
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                    if len(timesteps)>=1:
                        if config["ENV_NAME"]=="humanoidstandup":
                            for t in range(len(timesteps)):
                                writer.add_scalar('episodic return', return_values[t], timesteps[t])
                        else:
                            writer.add_scalar('episodic return', return_values[0], timesteps[0])
                    writer.add_scalar('Loss/total_loss', loss_info[0].mean(), step_)
                    writer.add_scalar('Loss/value_loss', loss_info[0][0].mean(), step_)
                    writer.add_scalar('Loss/actor_loss', loss_info[0][1].mean(), step_)
                    writer.add_scalar('Loss/entropy', loss_info[0][2].mean(), step_)

                jax.debug.callback(callback, info_total)

            runner_state = (train_state, env_state, last_obs, rng, current_index)
            return runner_state, metric
        current_index = 0
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obsv, _rng, current_index)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, length=int(config["NUM_UPDATES"])
        )
        writer.close()
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 3e-4,
        "NUM_ENVS": 2048,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 5e7,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 32,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.0,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "tanh",
        "ENV_NAME": "walker2d",
        "ANNEAL_LR": False,
        "NORMALIZE_ENV": True,
        "DEBUG": True,
        "FEATURE_DIM": 256,
        "INITIAL_LR": 1e-1,
        "UPDATE_PERCENT": 0.5,
        "NUM_STEPS_EVAL": 100,
        "CLIP_EPS_RATIO": 1,
        "RBF_RATE": 1e-0
        
    }
    rng = jax.random.PRNGKey(10)
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    