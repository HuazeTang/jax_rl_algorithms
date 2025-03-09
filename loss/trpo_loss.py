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
    print_grad_info
)

@dataclass
class LossConfig:
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
        # forward update network
        pi_new: distrax.MultivariateNormalDiag
        value: jax.Array
        pi_new, value = self.network.apply(params, batch.obs)
        log_prob_new = pi_new.log_prob(batch.action)
        
        # value loss (mse loss)
        value_loss = 0.5 * jnp.square(value - targets).mean()
        
        # policy grandient loss
        ratio = jnp.exp(log_prob_new - batch.log_prob)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor = -(ratio * gae).mean()
        
        # entropy regularization
        entropy = pi_new.entropy().mean()
        
        # calculate KL divergence between old policy and new policy
        kl_divergence = calculate_kl_divergence_with_clip(
            log_prob_old=batch.log_prob,
            log_prob_new=log_prob_new
        )
        
        # total loss
        # Note: optimization object of TRPO requires constraint; here it is only used for gradient calculation
        total_loss = (
            loss_actor 
            + self.config.vf_coef * value_loss 
            - self.config.ent_coef * entropy
        )
        
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
        """JAX 兼容的回溯线搜索"""
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
        
        # 生成候选 alpha 向量 [1.0, c, c^2, ..., c^k]
        backtrack_iters = self.config.backtrack_iters
        coeffs = jnp.ones(backtrack_iters + 1) * self.config.backtrack_coeff
        alphas = 1.0 * jnp.cumprod(coeffs)  # 向量化生成所有候选值
        
        # 并行计算所有候选 alpha 的 KL 散度
        kl_values = jax.vmap(line_search_loss)(alphas)
        
        # 寻找第一个满足 KL <= delta * 1.5 的 alpha
        delta_threshold = self.config.delta * 1.5
        valid_mask = (kl_values <= delta_threshold).astype(jnp.float32)
        
        # 若没有满足条件的 alpha，选择最后一个（最小步长）
        idx = jnp.argmax(valid_mask)  # 第一个满足条件的索引
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
        # 1. 计算损失和梯度
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)

        total_loss, grads = grad_fn(
            params, batch, advantages, targets
        )

        print_grad_info(grads, "Direct gradient")
        
        # 2. 用共轭梯度法求解自然梯度方向: F^{-1} * grad
        natural_grad = self.conjugate_gradient(params, batch, grads)

        print_grad_info(natural_grad, "Natural gradient")
        
        # 3. 计算未约束的步长方向
        step_direction = natural_grad
        
        # 4. 计算最大步长 beta，使得 beta^2 * (step_dir^T F step_dir) <= delta
        # calculate: (step_dir^T F step_dir)
        fvp_step = self.fisher_vector_product(params, batch, step_direction)
        shs = jax.tree_util.tree_reduce(
            lambda s, x: s + jnp.sum(x), 
            jax.tree_map(lambda s_leaf, f_leaf: s_leaf * f_leaf, step_direction, fvp_step), 
            0.0
        )
        beta = jnp.sqrt(self.config.delta / (shs + 1e-8))
        full_step = jax.tree_map(lambda s: beta * s, step_direction)
        
        # 5. back linear search for optimal alpha (step size)
        alpha = self.backtracking_line_search(params, full_step, batch)
        
        # 6. Assemble final gradient
        final_gradient = jax.tree_map(lambda x: alpha * x, full_step)

        print_grad_info(final_gradient, f"Final gradient with alpha {alpha}")
        
        return total_loss, final_gradient
    