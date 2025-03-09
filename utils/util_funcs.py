import jax
import jax.numpy as jnp
from jax import debug

# utils functions
def tree_dot(tree1, tree2):
    """calculate the dot product of two tree nodes"""
    return jax.tree_util.tree_reduce(
        lambda s, x: s + jnp.sum(x),
        jax.tree_map(lambda tree_1_node, tree_2_node: tree_1_node * tree_2_node, tree1, tree2),
        0.0
    )

def tree_squared_sum(tree):
    """calculate sum of square of all tree nodes"""
    return jax.tree_util.tree_reduce(
        lambda s, x: s + jnp.sum(x**2),
        tree,
        0.0
    )

def calculate_kl_divergence_with_clip(log_prob_old: jax.Array, log_prob_new: jax.Array):
    log_prob_new_clip = jnp.clip(log_prob_new, a_min=-1e8, a_max=1e8)
    log_prob_old_clip = jnp.clip(log_prob_old, a_min=-1e8, a_max=1e8)
    kl_divergence = jnp.mean(log_prob_old_clip - log_prob_new_clip)

    return kl_divergence

def print_grad_info(grads, info: str):
    """打印梯度的详细信息，包括结构、统计、异常值"""
    debug.print("\n" + "=" * 40 + f" 梯度分析 {info} " + "=" * 40)

    # 1. 打印梯度结构
    debug.print("\n[梯度结构]")
    print(jax.tree_map(lambda x: (x.shape, x.dtype), grads))

    # 2. 打印统计信息
    debug.print("\n[统计信息]")
    stats = jax.tree_map(
        lambda x: {
            "mean": jnp.mean(x).astype(float).item(),
            "max": jnp.max(x).astype(float).item(),
            "min": jnp.min(x).astype(float).item(),
            "std": jnp.std(x).astype(float).item()
        },
        grads
    )
    debug.print(str(stats))
    for k, v in stats.items():
        debug.print(f"{k}:")
        for kv, vv in v.items():
            debug.print(f"\t{kv}: {vv};")

    # 3. 检查NaN/Inf
    print("\n[异常值检查]")
    has_nan = jax.tree_map(lambda x: jnp.any(jnp.isnan(x)).item(), grads)
    has_inf = jax.tree_map(lambda x: jnp.any(jnp.isinf(x)).item(), grads)
    print("包含 NaN:", has_nan)
    print("包含 Inf:", has_inf)

    # 4. 打印路径信息（可选）
    print("\n[路径信息]")
    def _print_path(path, x):
        path_str = "/".join(map(str, path))
        print(f"Path: {path_str:<30} | Shape: {x.shape} | Dtype: {x.dtype}")
    
    jax.tree_util.tree_map_with_path(_print_path, grads)

    print("\n" + "=" * 85 + "\n")
