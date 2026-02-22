"""Drone control: yaw/roll from angle and x-offset with smoothing."""

from src.post_processing import exponential_smoothing
import numpy as np


def control_drone(mask, angle, x_offset, prev_yaw, prev_roll, alpha_yaw=0.3, alpha_roll=0.3, cam_fov=90):
    """Compute smoothed yaw and roll from angle and x-offset."""
    yaw = angle * 180 / np.pi
    smoothed_yaw = exponential_smoothing(yaw, prev_yaw, alpha_yaw)
    roll = (-x_offset) * cam_fov / mask.shape[1]
    smoothed_roll = exponential_smoothing(roll, prev_roll, alpha_roll)
    return float(smoothed_yaw), float(smoothed_roll)
