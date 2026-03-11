# Copyright 2026 Curious-VLA Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import math
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from navsim.common.dataclasses import AgentInput
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)

def _normalize_angle(angle: torch.Tensor) -> torch.Tensor:
    """Normalize angle tensor to [-pi, pi]."""
    return (angle + math.pi) % (2 * math.pi) - math.pi

def complete_trajectory(
    partial_trajectory: torch.Tensor,
    trajectory_sampling: TrajectorySampling
) -> torch.Tensor:
    """
    Complete or truncate a trajectory using a constant acceleration/velocity model
    to match the trajectory_sampling definition.

    Args:
        partial_trajectory: Partial or over-length trajectory from model output.
        trajectory_sampling: Config object with .num_poses and .interval_length.

    Returns:
        A trajectory tensor with the correct length.
    """
    target_steps = trajectory_sampling.num_poses
    dt = trajectory_sampling.interval_length

    current_steps = partial_trajectory.shape[0]

    if current_steps == target_steps:
        return partial_trajectory

    device = partial_trajectory.device
    dtype = partial_trajectory.dtype

    if current_steps > target_steps:
        return partial_trajectory[:target_steps]

    completed_trajectory = torch.zeros((target_steps, 3), device=device, dtype=dtype)
    completed_trajectory[:current_steps] = partial_trajectory

    if current_steps < 2:
        last_state = completed_trajectory[current_steps - 1]
        completed_trajectory[current_steps:] = last_state.expand(target_steps - current_steps, -1)
        return completed_trajectory

    p_last = completed_trajectory[current_steps - 1, :2]
    p_prev = completed_trajectory[current_steps - 2, :2]
    v_xy_last = (p_last - p_prev) / dt
    theta_last = completed_trajectory[current_steps - 1, 2]
    theta_prev = completed_trajectory[current_steps - 2, 2]
    omega_last = _normalize_angle(theta_last - theta_prev) / dt
    if current_steps < 3:
        a_xy_const = torch.zeros(2, device=device, dtype=dtype)
        alpha_const = torch.tensor(0.0, device=device, dtype=dtype)
    else:
        p_pprev = completed_trajectory[current_steps - 3, :2]
        v_xy_prev = (p_prev - p_pprev) / dt
        a_xy_const = (v_xy_last - v_xy_prev) / dt
        theta_pprev = completed_trajectory[current_steps - 3, 2]
        omega_prev = _normalize_angle(theta_prev - theta_pprev) / dt
        alpha_const = (omega_last - omega_prev) / dt
    for i in range(current_steps, target_steps):
        p_prev_step = completed_trajectory[i - 1, :2]
        theta_prev_step = completed_trajectory[i - 1, 2]
        time_elapsed = (i - current_steps) * dt
        v_xy_current = v_xy_last + a_xy_const * time_elapsed
        omega_current = omega_last + alpha_const * time_elapsed
        p_new = p_prev_step + v_xy_current * dt + 0.5 * a_xy_const * dt**2
        theta_new = theta_prev_step + omega_current * dt + 0.5 * alpha_const * dt**2
        completed_trajectory[i, :2] = p_new
        completed_trajectory[i, 2] = _normalize_angle(theta_new)
    logger.info(f"complete trajectory successfully, shape: {completed_trajectory.shape}")
    return completed_trajectory

def generate_constant_velocity_fallback(
    agent_input: AgentInput,
    trajectory_sampling: TrajectorySampling,
    device: str = 'cpu'
) -> torch.Tensor:
    """
    Generate a constant-velocity straight-line fallback trajectory
    based on agent_input and trajectory_sampling.
    """
    current_velocity = agent_input.ego_statuses[-1].ego_velocity[0].item()

    target_steps = trajectory_sampling.num_poses
    dt = trajectory_sampling.interval_length

    poses = torch.zeros((target_steps, 3), device=device, dtype=torch.float32)
    for i in range(target_steps):
        poses[i, 0] = current_velocity * (i + 1) * dt
    return poses
