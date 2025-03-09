from functools import partial
import jax
from jax import tree_util
import jax.numpy as jnp

def mixed_precision(compute_dtype=jnp.float16, param_dtype=jnp.float32, enabled=True):
    """decorator for mixed precision, autocast inputs/params/outputs"""
    def decorator(func):
        @partial(jax.jit, static_argnums=(0,))  # 确保JIT编译
        def wrapper(params, *args, **kwargs):
            if not enabled:
                return func(compute_params, *args, **kwargs)
            
            # 转换参数到计算精度
            compute_params = tree_util.tree_map(
                lambda x: x.astype(compute_dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x,
                params
            )
            # 转换输入数据到计算精度
            args = tree_util.tree_map(
                lambda x: x.astype(compute_dtype) if isinstance(x, jnp.ndarray) else x,
                args
            )
            # 执行前向计算
            outputs = func(compute_params, *args, **kwargs)
            # 转换输出回参数精度
            return tree_util.tree_map(
                lambda x: x.astype(param_dtype) if isinstance(x, jnp.ndarray) else x,
                outputs
            )
        return wrapper
    return decorator
