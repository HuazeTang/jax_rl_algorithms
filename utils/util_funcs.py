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

def print_jax_info(array_dict, info: str=""):
    """Print jax array/dict in a pretty way"""
    print("\n" + "=" * 40 + f" Jax array/dict: {info} " + "=" * 40)

    if isinstance(array_dict, jax.Array):
        array_dict = {"root": array_dict}

    def _print_path(path, x):
        path_str = "/".join(map(str, path))
        print(f"Path: {path_str:<45} | Shape: {x.shape} \t| Dtype: {x.dtype}")
    
    def _print_value(path, x):
        path_str = "/".join(map(str, path))
        if isinstance(x, jax.core.Tracer) or (isinstance(x, dict) and any(isinstance(v, jax.core.Tracer) for v in x.values())):
            print(f"Path: {path_str:<45}")
        else:
            print(f"Path: {path_str:<45} | Value: {x}")

    # 1. Print structure
    print("\n[Structure]")
    jax.tree_util.tree_map_with_path(_print_path, array_dict)

    # 2. Print statistics
    print("\n[Statistics]")
    stats = jax.tree_map(
        lambda x: {
            "mean": jnp.mean(x),
            "max": jnp.max(x),
            "min": jnp.min(x),
            "std": jnp.std(x)
        },
        array_dict
    )
    jax.tree_util.tree_map_with_path(_print_value, stats)

    # 3. Check NaN/Inf
    print("\n[Check NaN/Inf]")
    
    # Check NaN and Inf with tree_map
    has_nan = jax.tree_map(lambda x: jnp.any(jnp.isnan(x)), array_dict)
    all_nan = jax.tree_map(lambda x: jnp.all(jnp.isnan(x)), array_dict)
    has_inf = jax.tree_map(lambda x: jnp.any(jnp.isinf(x)), array_dict)
    all_inf = jax.tree_map(lambda x: jnp.all(jnp.isinf(x)), array_dict)

    # Define a function to print path if there is an anomaly, using jax.lax.cond to conditionally execute a function.
    def _print_path_if_anomaly(path, has_anomaly):
        path_str = "/".join(map(str, path))
        
        jax.lax.cond(
            has_anomaly,
            lambda _: debug.callback(lambda: print(f"Path: {path_str}")),  # 条件为真时执行
            lambda _: None,  # 条件为假时无操作
            operand=None
        )
        return None
    
    # Print paths with anomalies
    print("-----Path with NaN-----")
    jax.tree_util.tree_map_with_path(_print_path_if_anomaly, has_nan)

    print("-----Path all NaN-----")
    jax.tree_util.tree_map_with_path(_print_path_if_anomaly, all_nan)
    
    print("-----Path with Inf-----")
    jax.tree_util.tree_map_with_path(_print_path_if_anomaly, has_inf)

    print("-----Path all Inf-----")
    jax.tree_util.tree_map_with_path(_print_path_if_anomaly, all_inf)

    print("\n" + "=" * 85 + "\n")
