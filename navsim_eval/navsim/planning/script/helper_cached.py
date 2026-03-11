# helper_cached.py
import logging
import os
import time
from typing import Callable, Dict, List, Optional

import numpy as np

from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.metric_caching.metric_cache import MetricCache
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.traffic_agents_policies.abstract_traffic_agents_policy import AbstractTrafficAgentsPolicy

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calculate_pdms(score_dict: Dict) -> float:
    return (
        score_dict["no_at_fault_collisions"]
        * score_dict["drivable_area_compliance"]
        * (
            5 * score_dict["ego_progress"]
            + 5 * score_dict["time_to_collision_within_bound"]
            + 2 * score_dict["history_comfort"]
        )
        / 12
    )


def calculate_scaled_pdms(
    score_dict: Dict,
    gamma=(0.5, 0.5, 1.0),
    weights=(5, 5, 2),
) -> float:
    no_coll = score_dict.get("no_at_fault_collisions", 1.0)
    drive = score_dict.get("drivable_area_compliance", 1.0)
    if no_coll == 0 or drive == 0:
        return 0.0

    ego = np.clip(score_dict.get("ego_progress", 0.0), 0.0, 1.0)
    ttc = np.clip(score_dict.get("time_to_collision_within_bound", 0.0), 0.0, 1.0)
    comfort = np.clip(score_dict.get("history_comfort", 0.0), 0.0, 1.0)

    def focal_map(x: float, g: float) -> float:
        return 1.0 - (1.0 - x) ** g

    ego_f = focal_map(float(ego), float(gamma[0]))
    ttc_f = focal_map(float(ttc), float(gamma[1]))
    comfort_f = focal_map(float(comfort), float(gamma[2]))

    weighted_sum = weights[0] * ego_f + weights[1] * ttc_f + weights[2] * comfort_f
    total_weight = float(sum(weights))
    return float(np.clip(weighted_sum / total_weight, 0.0, 1.0))


def score_single_trajectory_cached(
    cfg: DictConfig,
    get_metric_cache: Callable[[str], MetricCache],
    token: str,
    poses: List[List[float]],
    verbose: bool = False,
) -> Dict:
    """Same scoring logic as helper.score_single_trajectory, but metric_cache is fetched via callback.

    This file is intentionally standalone so you can A/B test without touching existing code.
    """
    pid = os.getpid()
    t0 = time.perf_counter()
    # logger.info(f"[PID {os.getpid()}] scoring token={token}")
    print(f"[PID {os.getpid()}] scoring token={token}")
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)

    if cfg.traffic_agents == "non_reactive":
        traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
            cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
        )
    elif cfg.traffic_agents == "reactive":
        traffic_agents_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
    else:
        raise ValueError(f"Unknown traffic_agents mode: {cfg.traffic_agents}")

    t1 = time.perf_counter()
    metric_cache = get_metric_cache(token)
    t2 = time.perf_counter()

    trajectory = Trajectory(poses=np.array(poses))

    score_row, _ego_simulated_states = pdm_score(
        metric_cache=metric_cache,
        model_trajectory=trajectory,
        future_sampling=simulator.proposal_sampling,
        simulator=simulator,
        scorer=scorer,
        traffic_agents_policy=traffic_agents_policy,
    )

    t3 = time.perf_counter()

    score_dict = score_row.to_dict(orient="records")[0]
    score_dict["pdms"] = score_dict.get("pdm_score", calculate_pdms(score_dict))
    score_dict["pdms_scaled"] = calculate_scaled_pdms(score_dict)

    if verbose:
        score_dict["token"] = token
        score_dict["visual"] = ""
        score_dict["_pid"] = pid
        score_dict["_timing_s"] = {
            "instantiate": float(t1 - t0),
            "load_metric_cache": float(t2 - t1),
            "pdm_score": float(t3 - t2),
            "total": float(t3 - t0),
        }

    return score_dict


def score_same_token_trajectory_cached(
    cfg: DictConfig,
    metric_cache: MetricCache,
    poses: List[List[List[float]]],
    verbose: bool = False,
) -> List[Dict]:
    """Batch over poses for a single token (keeps token fixed, still no cross-request batching)."""

    pid = os.getpid()
    results: List[Dict] = []

    for pose in poses:
        simulator: PDMSimulator = instantiate(cfg.simulator)
        scorer: PDMScorer = instantiate(cfg.scorer)

        if cfg.traffic_agents == "non_reactive":
            traffic_agents_policy: AbstractTrafficAgentsPolicy = instantiate(
                cfg.traffic_agents_policy.non_reactive, simulator.proposal_sampling
            )
        elif cfg.traffic_agents == "reactive":
            traffic_agents_policy = instantiate(cfg.traffic_agents_policy.reactive, simulator.proposal_sampling)
        else:
            raise ValueError(f"Unknown traffic_agents mode: {cfg.traffic_agents}")

        trajectory = Trajectory(poses=np.array(pose))

        score_row, _ego_simulated_states = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
            traffic_agents_policy=traffic_agents_policy,
        )

        score_dict = score_row.to_dict(orient="records")[0]
        score_dict["pdms"] = score_dict.get("pdm_score", calculate_pdms(score_dict))
        score_dict["pdms_scaled"] = calculate_scaled_pdms(score_dict)

        if verbose:
            score_dict["visual"] = ""
            score_dict["_pid"] = pid

        results.append(score_dict)

    return results
