import jax
import jax.numpy as jnp
from functools import wraps

def check_nan_inf(x):
    """Check for NaN and Inf in JAX array or Python float."""
    def _check(value):
        if jnp.any(jnp.isnan(value)) or jnp.any(jnp.isinf(value)):
            raise ValueError("NaN or Inf detected!")
        return value

    # Apply jax.debug.callback to check for NaN and Inf in the input value.
    jax.debug.callback(_check, x)
    return x

def check_dict_nan_inf(data):
    """Check for NaN and Inf in JAX arrays or Python floats in a dictionary."""
    def _check_leaf(x):
        if isinstance(x, (jnp.ndarray, float)):  # Check for NaN and Inf in JAX array or Python float.
            check_nan_inf(x)
        return x

    # Check for NaN and Inf in JAX arrays or Python floats in a dictionary with jax.tree_map.
    jax.tree_map(_check_leaf, data)

def check_nan_inf_decorator(func):
    """Decorator to check for NaN and Inf in function output."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, dict):  # 检查输出是否为 dict
            check_dict_nan_inf(result)
        else:
            check_nan_inf(result)  # 如果不是 dict，直接检查
        return result
    return wrapper
