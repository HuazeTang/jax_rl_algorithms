import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
sys.path.insert(0, os.path.abspath(project_root))

import jax.numpy as jnp
from utils.util_funcs import print_jax_info

def test_print_jax_info():
    # 测试用例1: 简单数组
    grads_simple = jnp.array([1, 2, 3])
    print_jax_info(grads_simple, "Simple Array")

    # 测试用例2: 包含NaN值的数组
    grads_nan = jnp.array([1, jnp.nan, 3])
    print_jax_info(grads_nan, "Array with NaN")

    # 测试用例3: 嵌套字典，包含Inf值
    grads_nested = {
        "layer1": {
            "weights": jnp.array([jnp.inf, 2, 3]),
            "biases": jnp.array([1, 2, 3])
        },
        "layer2": jnp.array([1, 2, 3])
    }
    print_jax_info(grads_nested, "Nested Dict with Inf")

    # 测试用例4: 包含统计信息的复杂结构
    grads_complex = {
        "layer1": jnp.array([1.5, 2.5, 3.5]),
        "layer2": {
            "sublayer": jnp.array([-1, 0, 1])
        }
    }
    print_jax_info(grads_complex, "Complex Structure for Stats")

# 运行测试用例
test_print_jax_info()
