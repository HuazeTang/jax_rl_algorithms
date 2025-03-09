import math
from enum import Enum, auto
from typing import Callable, Optional, List

class ScheduleType(Enum):
    Linear = auto()
    Cosine = auto()
    Constant = auto()
    Combined = auto()

    def __str__(self) -> str:
        return self.name


def make_schedule(
    schedule_type: ScheduleType,
    max_lr: float,
    num_updates: int,
    num_total_mini_batches: int,
    schedule_list: Optional[List[ScheduleType]] = None,
    switch_points: Optional[List[float]] = None,
) -> Callable[[float], float]:
    """
    Factory function to create a learning rate schedule.

    Args:
        schedule_type (ScheduleEnum): Type of schedule.
        max_lr (float): Maximum learning rate.
        num_updates (int): Number of updates.
        num_total_mini_batches (int): Number of mini bathces in total.
        schedule_list (Optional[List[ScheduleEnum]]): List of schedule types for combined schedules.
        switch_points (Optional[List[ScheduleEnum]]): Points at which to switch between schedules.

    Returns:
        Callable[[float], float]: A function that takes the current step count and returns the learning rate.
    """
    # Define a dictionary mapping schedule types to their corresponding functions
    schedule_mapping = {
        ScheduleType.Linear: lambda count: max_lr * (1.0 - (count // num_total_mini_batches)) / num_updates,
        ScheduleType.Cosine: lambda count: max_lr * 0.5 * (1 + math.cos(math.pi * (count / (num_updates * num_total_mini_batches)))),
        ScheduleType.Constant: lambda count: max_lr,
    }

    if schedule_type == ScheduleType.Combined:
        if schedule_list is None or switch_points is None:
            raise ValueError("For 'combined' schedules, `schedule_list` and `switch_points` must be provided.")
        if len(schedule_list) != len(switch_points) + 1:
            raise ValueError("The number of switch points must be one less than the number of schedules.")

        # Create individual schedules using the mapping
        schedules = [schedule_mapping[s] for s in schedule_list]
        return CombinedSchedule(schedules, switch_points)

    elif schedule_type in schedule_mapping:
        return schedule_mapping[schedule_type]

    else:
        raise ValueError(f"Unsupported schedule type: {schedule_type}.")

class CombinedSchedule:
    """
    A schedule that combines multiple schedules sequentially.
    """

    def __init__(self, schedules: List[Callable[[float], float]], switch_points: List[float]):
        """
        Args:
            schedules (List[Callable[[float], float]]): List of schedule functions.
            switch_points (List[float]): Points at which to switch between schedules.
                                      The length should be `len(schedules) - 1`.
        """
        if len(schedules) != len(switch_points) + 1:
            raise ValueError("The number of switch points must be one less than the number of schedules.")
        self.schedules = schedules
        self.switch_points = switch_points

    def __call__(self, count: float) -> float:
        """
        Returns the learning rate for the given step count.
        """
        for i, switch_point in enumerate(self.switch_points):
            if count < switch_point:
                return self.schedules[i](count)
        return self.schedules[-1](count)
