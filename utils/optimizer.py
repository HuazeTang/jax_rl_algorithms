from enum import Enum, auto
from typing import Callable, Dict, Union, Optional
import optax

class OptimizerType(str, Enum):
    """Supported optimizer types."""
    NotUsing = auto()
    Adam = auto()
    SGD = auto()
    RMSprop = auto()
    AdamW = auto()

    def __str__(self) -> str:
        return self.name

# mapping from OptimizerType to optimizer constrcutor
OptimizerConstructor = Callable[..., optax.GradientTransformation]
OPTIMIZER_MAP: Dict[OptimizerType, OptimizerConstructor] = {
    OptimizerType.NotUsing: optax.sgd,
    OptimizerType.Adam: optax.adam,
    OptimizerType.SGD: optax.sgd,
    OptimizerType.RMSprop: optax.rmsprop,
    OptimizerType.AdamW: optax.adamw,
}

def init_tx(
    tx_type: OptimizerType,
    learning_rate: Union[float, Callable[[float], float]],
    max_grad_norm: float,
    eps: float = 1e-8,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    use_mixed_precision: bool = False,
) -> optax.GradientTransformation:
    """
    Create a gradient transformation chain based on configuration.
    
    Args:
        tx_type (OptimizerType): Type of optimizer.
        learning_rate (Union[float, Callable[[float], float]]): Learning rate setting or schedule
        max_grad_norm (float): Gradient norm clip threshold. 
        eps (float): EPS value for Adam/RMSprop, default as 0.9.
        momentum (float): Momentum for SGD/RMSprop, default as 0.9
        weight_decay (float): Weight decay for AdamW, default as 0
        use_mixed_precision (bool): Whether use mixed precision or not
        
    Returns:
        optax.GradientTransformation: Optimizer chain with clipping.
    """
    # 1. grad norm clip (always as the first step)
    tx_chain = [optax.clip_by_global_norm(max_grad_norm)]

    # TODO: deal with mixed precision
    # if use_mixed_precision:
    #     if loss_scale is None:
    #         loss_scale = 2**10 # typical init number
        
    #     # add 
    #     tx_chain.append(
    #         optax.(loss_scale=loss_scale)
    #     )
    
    # 2. get optimizer constructor function
    optimizer_constructor = OPTIMIZER_MAP.get(tx_type)
    if optimizer_constructor is None:
        raise ValueError(f"Unsupported optimizer type: {tx_type}. Valid options: {list(OPTIMIZER_MAP.keys())}")
    
    # 3. bulid optimizer params
    optimizer_kwargs = {}

    if tx_type == OptimizerType.NotUsing:
        return optax.chain(
            optax.scale_by_learning_rate(learning_rate),  # 学习率保留在计算图中
        )
    
    # Adam/RMSprop requires eps
    if tx_type in (OptimizerType.Adam, OptimizerType.RMSprop):
        optimizer_kwargs["eps"] = eps
    
    # SGD/RMSprop requires momentum
    if tx_type in (OptimizerType.SGD, OptimizerType.RMSprop):
        optimizer_kwargs["momentum"] = momentum
    
    # AdamW requires weight_decay
    if tx_type == OptimizerType.AdamW:
        optimizer_kwargs["weight_decay"] = weight_decay
    
    # 4. create optimizer
    optimizer = optimizer_constructor(
        learning_rate=learning_rate,
        **optimizer_kwargs
    )
    
    tx_chain.append(optimizer)
    return optax.chain(*tx_chain)
