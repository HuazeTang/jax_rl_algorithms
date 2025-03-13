import jax
import jax.numpy as jnp
from dataclasses import dataclass
import distrax
import flax.linen as nn
from typing import Tuple, Dict
from utils.transition import Transition
from utils.util_funcs import (
    tree_dot, 
    tree_squared_sum, 
    calculate_kl_divergence_with_clip, 
    print_jax_info
)

@dataclass
class LossConfig:
    clip_eps: float
    delta: float = 0.01  # KL divergence
    cg_iters: int = 10   # 共轭梯度法迭代次数
    backtrack_iters: int = 15 # 回溯线搜索迭代次数
    backtrack_coeff: float = 1. # 步长衰减系数
    vf_coef: float = 0.5  # 值函数损失系数
    ent_coef: float = 0.0 # 熵正则化系数


class TRPOLoss:
    def __init__(self, network: nn.Module, config: LossConfig):
        self.network = network
        self.config = config

    def compute_loss(
        self, 
        params: dict, 
        batch: Transition,
        gae: jax.Array, 
        targets: jax.Array
    ) -> Tuple[jax.Array, Dict[str, jax.Array]]:
        """Calculate basic loss and necessary metrics (e.g. KL divergence, entropy, etc.)"""
        # forward update networkclip_eps: float
        pi_new: distrax.MultivariateNormalDiag
        value: jax.Array
        pi_new, value = self.network.apply(params, batch.obs)
        log_prob_new = pi_new.log_prob(batch.action)
        
        # value loss (mse loss)
        value_loss = 0.5 * jnp.square(value - targets).mean()
        
        # policy grandient loss
        log_prob_diff = log_prob_new - batch.log_prob
        log_prob_diff = jnp.clip(log_prob_diff, -1e8, 1e8)
        ratio = jnp.exp(log_prob_diff)
        # print_jax_info({"ratio": ratio}, "ratio")
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor = -(jnp.clip(ratio, 1-self.config.clip_eps, 1+self.config.clip_eps) * gae).mean()
        # print_jax_info(loss_actor, "loss actor")
        
        # entropy regularization
        entropy = pi_new.entropy().mean()
        
        # calculate KL divergence between old policy and new policy
        kl_divergence = calculate_kl_divergence_with_clip(
            log_prob_old=batch.log_prob,
            log_prob_new=log_prob_new
        )
        # print_jax_info(kl_divergence, "kl divergence")
        
        # total loss
        # Note: optimization object of TRPO requires constraint; here it is only used for gradient calculation
        total_loss = (
            # loss_actor 
            + self.config.vf_coef * value_loss 
            # - self.config.ent_coef * entropy
        )
        
        # print_jax_info(total_loss, "total loss")

        stats = {
            'loss/total': total_loss,
            'loss/actor': loss_actor,
            'loss/value': value_loss,
            'stats/entropy': entropy,
            'stats/kl': kl_divergence,
            'stats/ratio': ratio.mean(),
        }
        return total_loss, stats

    def fisher_vector_product(
        self, 
        params: dict, 
        batch: Transition, 
        vector: dict
    ) -> dict:
        """
        Calculate Fisher information matrix and vector product. 
        High efficiency of Hessian-vector product with auto-grad.
        """
        # Calculate gradient of kl divergence
        def kl_divergence(params):
            pi: distrax.MultivariateNormalDiag
            pi, _ = self.network.apply(params, batch.obs)
            log_prob = pi.log_prob(batch.action)
            return calculate_kl_divergence_with_clip(
                log_prob_old=batch.log_prob,
                log_prob_new=log_prob
            )
        
        # Calculate gradient
        grad_kl = jax.grad(kl_divergence)(params)
        
        # Calculate gradient and dot product with vector v
        dot_product = sum(
            jnp.sum(g * v) for g, v in zip(jax.tree_util.tree_leaves(grad_kl), 
                                         jax.tree_util.tree_leaves(vector))
        )
        
        # Calculate Hessian-vector product (via gradient of product of gradient)
        fvp = jax.grad(lambda p: dot_product)(params)

        return fvp

    def conjugate_gradient(
        self, 
        params: dict, 
        batch: Transition, 
        b: dict,
    ) -> dict:
        """Conjugate gradient to solve F^(-1) * b, where F is Fisher matrix"""
        cg_iters = self.config.cg_iters
            
        def body_fn(val):
            x, r, p, rdotr = val
            fvp = self.fisher_vector_product(params, batch, p)
            shs = tree_dot(p, fvp)
            alpha = rdotr / (shs + 1e-8)
            x = jax.tree_map(lambda x_, p_, a=alpha: x_ + a * p_, x, p)
            r = jax.tree_map(lambda r_, fvp_, a=alpha: r_ - a * fvp_, r, fvp)
            new_rdotr = tree_squared_sum(r)
            beta = new_rdotr / rdotr
            p = jax.tree_map(lambda r_, p_, b=beta: r_ + b * p_, r, p)
            return x, r, p, new_rdotr
        
        # init values
        x = jax.tree_map(jnp.zeros_like, b)
        r = jax.tree_map(jnp.copy, b)
        p = jax.tree_map(jnp.copy, r)
        rdotr_init = tree_squared_sum(r)
        
        # iterative updates
        val_init = (x, r, p, rdotr_init)
        val = jax.lax.fori_loop(0, cg_iters, lambda i, val: body_fn(val), val_init)
        x, _, _, _ = val
        return x
    
    def backtracking_line_search(
        self, 
        params: dict, 
        full_step: dict, 
        batch: Transition
    ) -> float:
        """Line search to find proper step size"""
        def line_search_loss(alpha):
            new_params = jax.tree_map(
                lambda p, s: p + alpha * s, 
                params, 
                full_step
            )
            pi_new: distrax.MultivariateNormalDiag
            pi_new, _ = self.network.apply(new_params, batch.obs)
            log_prob_new = pi_new.log_prob(batch.action)
            kl = calculate_kl_divergence_with_clip(
                log_prob_old=batch.log_prob,
                log_prob_new=log_prob_new
            )
            return kl
        
        # Generate a list of candidate alpha values
        backtrack_iters = self.config.backtrack_iters
        coeffs = jnp.ones(backtrack_iters + 1) * self.config.backtrack_coeff
        alphas = 1.0 * jnp.cumprod(coeffs)  # Vectorize generation of all candidate values
        
        # Get KL divergence of all candidate values in parallel
        kl_values = jax.vmap(line_search_loss)(alphas)
        
        # Get the first valid alpha satisfying KL <= delta
        delta_threshold = self.config.delta
        valid_mask = (kl_values <= delta_threshold).astype(jnp.float32)
        
        # If there is no valid alpha, return the last one
        idx = jnp.argmax(valid_mask)  # The first index of the last valid alpha
        idx = jnp.where(valid_mask.sum() > 0, idx, backtrack_iters)
        
        return alphas[idx]

    def update_step(
        self,
        params: dict,
        batch: Transition,
        advantages: jax.Array,
        targets: jax.Array,
    ) -> Tuple[dict, dict]:
        """Parameters update with TRPO"""
        # print_jax_info(advantages, "Advantages")

        # 1. get total loss and gradient
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)

        total_loss, grads = grad_fn(
            params, batch, advantages, targets
        )

        # print_jax_info(total_loss, "Loss")
        # print_jax_info(grads, "Direct gradient")
        
        # 2. solve natural gradient direction F^{-1} * grad with conjugate gradient
        natural_grad = self.conjugate_gradient(params, batch, grads)

        # print_jax_info(natural_grad, "Natural gradient")
        
        # 3. get step direction without constraint
        step_direction = natural_grad
        
        # 4. get the maximal step length beta to make beta^2 * (step_dir^T F step_dir) <= delta
        # 4.1 calculate: (step_dir^T F step_dir)
        fvp_step = self.fisher_vector_product(params, batch, step_direction)
        shs = jax.tree_util.tree_reduce(
            lambda s, x: s + jnp.sum(x), 
            jax.tree_map(lambda s_leaf, f_leaf: s_leaf * f_leaf, step_direction, fvp_step), 
            0.0
        )
        # 4.2 calculate beta
        beta = jnp.sqrt(self.config.delta / (shs + 1e-8))
        full_step = jax.tree_map(lambda s: beta * s, step_direction)
        
        # 5. back linear search for optimal alpha (step size)
        alpha = self.backtracking_line_search(params, full_step, batch)
        
        # 6. assemble final gradient
        final_gradient = jax.tree_map(lambda x: alpha * x, full_step)

        # print_jax_info(final_gradient, f"Final gradient with alpha {alpha}")
        
        return total_loss, final_gradient
    