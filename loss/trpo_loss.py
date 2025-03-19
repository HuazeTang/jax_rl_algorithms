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
from utils.debug_tools import check_nan_inf_decorator

@dataclass
class LossConfig:
    clip_eps: float
    delta: float = 0.01  # KL divergence
    cg_iters: int = 100   # 共轭梯度法迭代次数
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
        pi_new, value = self.network.apply(params, jax.lax.stop_gradient(batch.obs))
        log_prob_new = pi_new.log_prob(jax.lax.stop_gradient(batch.action))
        
        # value loss (mse loss)
        targets = jax.lax.stop_gradient(targets)
        value_loss = 0.5 * jnp.square(value - targets).mean()
        
        # policy grandient loss
        log_prob_diff = log_prob_new - jax.lax.stop_gradient(batch.log_prob)
        log_prob_diff = jnp.clip(log_prob_diff, -1e8, 1e8)
        ratio = jnp.exp(log_prob_diff)

        gae = jax.lax.stop_gradient(gae)
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor = -(ratio * gae).mean()
        # loss_actor = -(jnp.clip(ratio, 1-self.config.clip_eps, 1+self.config.clip_eps) * gae).mean()
        
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

    @check_nan_inf_decorator
    def fisher_vector_product(
        self, 
        params: dict, 
        batch: Transition, 
        vector: dict
    ) -> dict:
        """
        Calculate Fisher-vector product \(F * v\), where \(F\) is the Fisher information matrix.

        The Fisher-vector product is computed efficiently using automatic differentiation,
        avoiding the explicit construction of the Fisher matrix.

        Args:
            params: Model parameters (a PyTree).
            batch: A batch of transitions (data used to compute the Fisher matrix).
            vector: The vector \(v\) to multiply with the Fisher matrix (a PyTree).

        Returns:
            fvp: The Fisher-vector product \(F * v\) (a PyTree).
        """
        # Define the KL divergence function
        def kl_divergence(params):
            """
            Compute the KL divergence between the old and new policy distributions.

            Args:
                params: Model parameters (a PyTree).

            Returns:
                kl: The KL divergence (a scalar).
            """
            # Apply the policy network to get the new action distribution
            pi: distrax.MultivariateNormalDiag
            pi, _ = self.network.apply(params, jax.lax.stop_gradient(batch.obs))

            # Compute the log probability of actions under the new policy
            log_prob = pi.log_prob(jax.lax.stop_gradient(batch.action))

            # Calculate the KL divergence with clipping (for numerical stability)
            return calculate_kl_divergence_with_clip(
                log_prob_old=jax.lax.stop_gradient(batch.log_prob),
                log_prob_new=log_prob
            )
        
        # Calculate gradient
        grad_kl = jax.grad(kl_divergence)(params)
        
        # Compute the dot product between the gradient of KL and the input vector \(v\)
        vector = jax.lax.stop_gradient(vector)
        dot_product = sum(
            jnp.sum(g * v) for g, v in zip(jax.tree_util.tree_leaves(grad_kl), 
                                         jax.tree_util.tree_leaves(vector))
        )
        
        # Compute the Fisher-vector product \(F * v\) as the gradient of the dot product
        # This is equivalent to the Hessian-vector product of the KL divergence
        fvp = jax.grad(lambda p: dot_product)(params)
        
        # Prevent gradient propagation
        fvp = jax.lax.stop_gradient(fvp)

        return fvp

    @check_nan_inf_decorator
    def conjugate_gradient(
        self, 
        params: dict, 
        batch: Transition, 
        b: dict,
    ) -> dict:
        """Conjugate gradient to solve F^{-1} * b, where F is the Fisher matrix.

        Args:
            params: Model parameters (a PyTree).
            batch: A batch of transitions (data used to compute the Fisher matrix).
            b: The target vector (a PyTree).

        Returns:
            x: The solution to F^{-1} * b (a PyTree).
        """
        # Number of conjugate gradient iterations
        cg_iters = self.config.cg_iters
            
        def body_fn(val):
            """Body function for conjugate gradient iterations.

            Args:
                val: A tuple containing:
                    - x: Current solution estimate.
                    - r: Current residual.
                    - p: Current search direction.
                    - rdotr: Dot product of residual with itself.

            Returns:
                Updated values of x, r, p, and rdotr.
            """
            x, r, p, rdotr = val

            # Compute the Fisher vector product (fvp = F * p) using the Fisher matrix.
            fvp = self.fisher_vector_product(params, batch, p)
            jax.debug.callback(lambda q: print_jax_info(q, "fvp"), fvp)
            # Compute the scalar product of the residual (shs = p^T * F * p) with the fvp.
            shs = tree_dot(p, fvp)
            # Compute step size \alpha = \frac{r^T r}{p^T F p}
            alpha = rdotr / (shs + 1e-8) # Small constant for numerical stability
            # Update solution: x = x + \alpha * p
            x = jax.tree_map(lambda x_, p_, a=alpha: x_ + a * p_, x, p)
            # Update residual: r = r - \alpha * F * p
            r = jax.tree_map(lambda r_, fvp_, a=alpha: r_ - a * fvp_, r, fvp)
            # Compute new residual squared norm: r^T r
            new_rdotr = tree_squared_sum(r)
            # Compute \beta = \frac{r_{\text{new}}^T r_{\text{new}}}{r^T r}
            beta = new_rdotr / (rdotr + 1e-8) # Small constant for numerical stability
            # Update search direction: p = r + \beta * p
            p = jax.tree_map(lambda r_, p_, b=beta: r_ + b * p_, r, p)
            return x, r, p, new_rdotr
        
        # init values
        x = jax.tree_map(jnp.zeros_like, b) # x: Initial solution estimate (zeros)
        r = jax.tree_map(jnp.copy, b)       # r: Initial residual (r = b - F * x, but x=0 so r = b)
        p = jax.tree_map(jnp.copy, r)       # p: Initial search direction (p = r)
        rdotr_init = tree_squared_sum(r)    # rdotr: Initial residual squared norm (\(r^T r\))

        # stop init values gradient
        x = jax.lax.stop_gradient(x)
        r = jax.lax.stop_gradient(r)
        p = jax.lax.stop_gradient(p)
        rdotr_init = jax.lax.stop_gradient(rdotr_init)
        
        # iterative updates
        val_init = (x, r, p, rdotr_init)

        # run conjugate gradient iterations
        val = jax.lax.fori_loop(0, cg_iters, lambda i, val: body_fn(val), val_init)
        x, _, _, _ = val

        # Stop gradient to prevent backpropagation through the solver
        x = jax.lax.stop_gradient(x)
        jax.debug.callback(lambda x: print_jax_info(x, "cg gradient"), x)

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

        # 1. get total loss and gradient
        grad_fn = jax.value_and_grad(self.compute_loss, has_aux=True)

        total_loss, grads = grad_fn(
            params, batch, advantages, targets
        )
        
        # # 2. solve natural gradient direction F^{-1} * grad with conjugate gradient
        natural_grad = self.conjugate_gradient(params, batch, grads)
        
        # # 3. get step direction without constraint
        step_direction = natural_grad
        
        # # 4. get the maximal step length beta to make beta^2 * (step_dir^T F step_dir) <= delta
        # # 4.1 calculate: (step_dir^T F step_dir)
        # fvp_step = self.fisher_vector_product(params, batch, step_direction)
        # shs = jax.tree_util.tree_reduce(
        #     lambda s, x: s + jnp.sum(x), 
        #     jax.tree_map(lambda s_leaf, f_leaf: s_leaf * f_leaf, step_direction, fvp_step), 
        #     0.0
        # )
        # # 4.2 calculate beta
        # beta = jnp.sqrt(self.config.delta / (shs + 1e-8))
        beta = 1.0
        full_step = jax.tree_map(lambda s: beta * s, step_direction)
        
        # # 5. back linear search for optimal alpha (step size)
        # alpha = self.backtracking_line_search(params, full_step, batch)
        alpha = 0.01
        
        # # 6. assemble final gradient
        final_gradient = jax.tree_map(lambda x: alpha * x, full_step)
        
        return total_loss, final_gradient
    