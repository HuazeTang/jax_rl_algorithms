import json
import yaml
import toml
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable

from .schedule import ScheduleType, make_schedule
from .optimizer import OptimizerType, init_tx

@dataclass
class AlgoConfig:
    """
    A configuration class for reinforcement learning algorithm hyperparameters and settings.
    """
    # Learning process related
    NUM_ENVS: int        # Number of parallel environments.
    NUM_STEPS: int       # Number of steps to run in each environment per policy update.
    TOTAL_TIMESTEPS: int # Total number of timesteps for training.
    UPDATE_EPOCHS: int   # Number of epochs to update the policy with the same data.
    NUM_MINIBATCHES: int # Number of minibatches to split the data into for updates.

    # Loss related
    GAMMA: float      # Discount factor for future rewards.
    GAE_LAMBDA: float # Lambda parameter for Generalized Advantage Estimation (GAE).
    CLIP_EPS: float   # Epsilon for clipping in the PPO objective (surrogate loss).
    ENT_COEF: float   # Coefficient for entropy regularization term.
    VF_COEF: float    # Coefficient for value function loss.
    
    # Optimizer related
    LR: float              # Learning rate for the optimizer.
    MAX_GRAD_NORM: float   # Maximum gradient norm for gradient clipping. 
    TX_TYPE: OptimizerType # Type of optimizer used in optimizer generation.

    # Model related
    ACTIVATION: str # Activation function to use in the neural network (e.g., 'relu', 'tanh').
    
    # Schedule related
    ANNEAL_LR: bool             # Whether to anneal the learning rate over time.
    SCHEDULE_TYPE: ScheduleType # Type of schedule used in anneal learning rate
    
    # Envrionment related
    ENV_NAME: str        # Name of the environment to train on.
    NORMALIZE_ENV: bool  # Whether to normalize environment observations and rewards.
    
    # Logging and debug
    DEBUG: bool # Whether to enable debug mode (e.g., additional logging or checks).

    # Learning process related with None init
    NUM_UPDATES: Optional[int] = None    # Number of updates to perform during training. Automatically calculated.
    MINIBATCH_SIZE: Optional[int] = None # Size of each minibatch. Automatically calculated.

    # Optimizer related with None init
    SCHEDULE_LISTS: Optional[List[ScheduleType]] = None # List of schedule types for combined schedules.
    SWITCH_POINTS: Optional[List[float]] = None         # Points at which to switch between schedules.

    # Optimizer related with None init
    EPS: float = 1e-5          # EPS value for Adam/RMSprop optimzier
    MOMENTUM: float = 0.9      # Momentum value for SGD/RMSprop
    WEIGHT_DECAY: float = 0.0  # Weight decay value for AdamW

    def __post_init__(self):
        """
        Automatically calculates derived attributes after initialization.
        - NUM_UPDATES: Total number of updates based on timesteps, steps, and environments.
        - MINIBATCH_SIZE: Size of each minibatch based on environments, steps, and minibatches.
        """
        self.NUM_UPDATES = self.TOTAL_TIMESTEPS // self.NUM_STEPS // self.NUM_ENVS
        self.MINIBATCH_SIZE = self.NUM_ENVS * self.NUM_STEPS // self.NUM_MINIBATCHES

        if self.SCHEDULE_TYPE == ScheduleType.Combined and (
            self.SWITCH_POINTS is not None or self.SCHEDULE_LISTS is not None
        ):
            raise Warning(f"SWITCH_POINTS and SCHEDULE_LISTS should be None if not using Combined schedule")

def read_config_from_file(file_path: str) -> AlgoConfig:
    """
    Reads configuration settings from a file (JSON, YAML, or TOML) and returns an AlgoConfig instance.

    Args:
        file_path (str): Path to the configuration file.

    Returns:
        AlgoConfig: An instance of the AlgoConfig class populated with the configuration settings.

    Raises:
        ValueError: If the file format is not supported or the file is invalid.
    """
    # Determine the file format based on the file extension
    if file_path.endswith(".json"):
        with open(file_path, "r") as f:
            config_data: Dict[str, Any] = json.load(f)
    elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
        with open(file_path, "r") as f:
            config_data = yaml.safe_load(f)
    elif file_path.endswith(".toml"):
        with open(file_path, "r") as f:
            config_data = toml.load(f)
    else:
        raise ValueError("Unsupported file format. Use JSON, YAML, or TOML.")
    
    if "SCHEDULE_TYPE" in config_data:
        config_data["SCHEDULE_TYPE"] = ScheduleType[config_data["SCHEDULE_TYPE"]]

    # Create and return an instance of AlgoConfig
    return AlgoConfig(**config_data)

def make_schedule_from_config(config: AlgoConfig):
    if config.ANNEAL_LR:
        schedule_type = config.SCHEDULE_TYPE
    else:
        schedule_type = ScheduleType.Constant

    schedule = make_schedule(
        schedule_type=schedule_type,
        max_lr=config.LR,
        num_updates=config.NUM_UPDATES,
        num_total_mini_batches=config.NUM_MINIBATCHES*config.UPDATE_EPOCHS,
        schedule_list=config.SCHEDULE_LISTS,
        switch_points=config.SWITCH_POINTS
    )

    return schedule

def make_tx_from_config(config: AlgoConfig, schedule_fn: Callable):
    tx = init_tx(
        tx_type=config.TX_TYPE,
        learning_rate=schedule_fn,
        max_grad_norm=config.MAX_GRAD_NORM,
        eps=config.EPS,
        momentum=config.MOMENTUM,
        weight_decay=config.WEIGHT_DECAY
    )

    return tx
