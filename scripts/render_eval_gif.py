"""Render navsim evaluation results (predicted vs human GT trajectories) as GIFs.

Reads the saved PDM score CSV and `detailed_logs.jsonl` produced by
`run_pdm_score_one_stage.py`, and renders per-token GIFs with the front camera
beside a BEV panel that overlays both the human GT future trajectory and the
agent's cached predicted trajectory.

Examples
--------
# render a single token
python3 scripts/render_eval_gif.py \
    --exp_dir exp_root/curious_vla_qwen2_5_vl_3b_sft_stage2/2026.04.21.13.16.37 \
    --data_root datasets/navsim_test --split test \
    --token df5915c3464e569e

# render the 10 worst-scoring tokens by `score`
python3 scripts/render_eval_gif.py \
    --exp_dir exp_root/curious_vla_qwen2_5_vl_3b_sft_stage2/2026.04.21.13.16.37 \
    --data_root datasets/navsim_test --split test \
    --topk_worst 10 --metric score
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "navsim_eval"))


def _bootstrap_env(data_root: Path) -> None:
    """Set env vars consumed at import time by navsim / nuplan map APIs."""
    os.environ.setdefault("NUPLAN_MAP_VERSION", "nuplan-maps-v1.0")
    os.environ["OPENSCENE_DATA_ROOT"] = str(data_root)
    os.environ["NUPLAN_MAPS_ROOT"] = str(data_root / "maps")


def _parse_data_root() -> Path:
    """Pre-parse --data_root before heavy imports (env must be set pre-import)."""
    for i, tok in enumerate(sys.argv):
        if tok == "--data_root" and i + 1 < len(sys.argv):
            return Path(sys.argv[i + 1]).resolve()
        if tok.startswith("--data_root="):
            return Path(tok.split("=", 1)[1]).resolve()
    return (PROJECT_ROOT / "datasets/navsim_test").resolve()


_bootstrap_env(_parse_data_root())


import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from nuplan.common.actor_state.state_representation import StateSE2  # noqa: E402
from nuplan.common.geometry.convert import relative_to_absolute_poses  # noqa: E402
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling  # noqa: E402

from navsim.common.dataclasses import (  # noqa: E402
    Annotations,
    Cameras,
    EgoStatus,
    Frame,
    Lidar,
    Scene,
    SceneFilter,
    SceneMetadata,
    SensorConfig,
    Trajectory,
)
from navsim.common.dataloader import SceneLoader  # noqa: E402
from navsim.planning.simulation.planner.pdm_planner.utils.pdm_geometry_utils import (  # noqa: E402
    convert_absolute_to_relative_se2_array,
)
from navsim.visualization.bev import (  # noqa: E402
    add_configured_bev_on_ax,
    add_trajectory_to_bev_ax,
)

# Patch: the stock Scene.from_scene_dict_list unconditionally wraps lidar_path in Path()
# even when it is None for some frames, which raises TypeError during scene loading.
# We only need cameras + map + annotations, so install a lidar-tolerant replacement.
from pyquaternion import Quaternion


@classmethod
def _patched_scene_from_scene_dict_list(  # type: ignore[misc]
    cls,
    scene_dict_list,
    sensor_blobs_path,
    num_history_frames,
    num_future_frames,
    sensor_config,
):
    assert len(scene_dict_list) >= 0, "Scene list is empty!"
    scene_metadata = SceneMetadata(
        log_name=scene_dict_list[num_history_frames - 1]["log_name"],
        scene_token=scene_dict_list[num_history_frames - 1]["scene_token"],
        map_name=scene_dict_list[num_history_frames - 1]["map_location"],
        initial_token=scene_dict_list[num_history_frames - 1]["token"],
        num_history_frames=num_history_frames,
        num_future_frames=num_future_frames,
    )
    map_api = cls._build_map_api(scene_metadata.map_name)

    frames = []
    for frame_idx in range(len(scene_dict_list)):
        global_ego_status = cls._build_ego_status(scene_dict_list[frame_idx])
        annotations = cls._build_annotations(scene_dict_list[frame_idx])

        sensor_names = sensor_config.get_sensors_at_iteration(frame_idx)

        cameras = Cameras.from_camera_dict(
            sensor_blobs_path=sensor_blobs_path,
            camera_dict=scene_dict_list[frame_idx]["cams"],
            sensor_names=sensor_names,
        )

        raw_lidar_path = scene_dict_list[frame_idx].get("lidar_path")
        if raw_lidar_path is None:
            lidar = Lidar()
        else:
            from pathlib import Path as _Path

            lidar = Lidar.from_paths(
                sensor_blobs_path=sensor_blobs_path,
                lidar_path=_Path(raw_lidar_path),
                sensor_names=sensor_names,
            )

        frames.append(
            Frame(
                token=scene_dict_list[frame_idx]["token"],
                timestamp=scene_dict_list[frame_idx]["timestamp"],
                roadblock_ids=scene_dict_list[frame_idx]["roadblock_ids"],
                traffic_lights=scene_dict_list[frame_idx]["traffic_lights"],
                annotations=annotations,
                ego_status=global_ego_status,
                lidar=lidar,
                cameras=cameras,
            )
        )

    return Scene(scene_metadata=scene_metadata, map_api=map_api, frames=frames)


Scene.from_scene_dict_list = _patched_scene_from_scene_dict_list  # type: ignore[assignment]

from navsim.visualization.camera import add_camera_ax  # noqa: E402
from navsim.visualization.config import (  # noqa: E402
    TRAJECTORY_CONFIG,
)
from navsim.visualization.plots import configure_bev_ax  # noqa: E402


TRAJECTORY_SAMPLING = TrajectorySampling(time_horizon=4, interval_length=0.5)


def _trajectory_from_poses(poses: np.ndarray) -> Optional[Trajectory]:
    """Build a NAVSIM trajectory with a sampling length matching the pose count."""
    if len(poses) == 0:
        return None
    return Trajectory(
        poses.astype(np.float32),
        TrajectorySampling(num_poses=len(poses), interval_length=TRAJECTORY_SAMPLING.interval_length),
    )


def _absolute_predicted_poses(scene, predicted_poses: np.ndarray) -> np.ndarray:
    """Convert cached prediction from token-local coordinates to global poses."""
    current_idx = scene.scene_metadata.num_history_frames - 1
    origin = StateSE2(*scene.frames[current_idx].ego_status.ego_pose)
    relative_states = [StateSE2(*pose) for pose in predicted_poses.astype(np.float64)]
    absolute_states = relative_to_absolute_poses(origin, relative_states)
    return np.asarray([[state.x, state.y, state.heading] for state in absolute_states], dtype=np.float64)


def _relative_trajectory_from_absolute(absolute_poses: np.ndarray, origin_pose: np.ndarray) -> Optional[Trajectory]:
    """Convert global poses into a trajectory in the displayed frame's ego coordinates."""
    if len(absolute_poses) == 0:
        return None
    relative_poses = convert_absolute_to_relative_se2_array(
        StateSE2(*origin_pose),
        absolute_poses.astype(np.float64),
    )
    return _trajectory_from_poses(relative_poses)


def _human_future_trajectory_for_frame(scene, frame_idx: int) -> Optional[Trajectory]:
    """Return GT future trajectory relative to the displayed frame."""
    end_idx = min(frame_idx + 1 + TRAJECTORY_SAMPLING.num_poses, len(scene.frames))
    future_poses = np.asarray(
        [scene.frames[idx].ego_status.ego_pose for idx in range(frame_idx + 1, end_idx)],
        dtype=np.float64,
    )
    return _relative_trajectory_from_absolute(future_poses, scene.frames[frame_idx].ego_status.ego_pose)


def _agent_trajectory_for_frame(scene, absolute_predicted_poses: np.ndarray, frame_idx: int) -> Optional[Trajectory]:
    """Return the remaining cached prediction relative to the displayed frame."""
    current_idx = scene.scene_metadata.num_history_frames - 1
    start_idx = max(0, frame_idx - current_idx)
    remaining_predicted = absolute_predicted_poses[start_idx:]
    return _relative_trajectory_from_absolute(remaining_predicted, scene.frames[frame_idx].ego_status.ego_pose)


def _add_front_camera(ax: plt.Axes, frame) -> None:
    cam = frame.cameras.cam_f0
    if cam is None or cam.image is None:
        ax.text(0.5, 0.5, "no cam_f0", ha="center", va="center")
    else:
        add_camera_ax(ax, cam)


def _plot_front_and_bev(fig, ax, scene, frame_idx: int, absolute_predicted_poses: np.ndarray) -> None:
    frame = scene.frames[frame_idx]
    _add_front_camera(ax[0], frame)

    add_configured_bev_on_ax(ax[1], scene.map_api, frame)

    human_trajectory = _human_future_trajectory_for_frame(scene, frame_idx)
    if human_trajectory is not None:
        add_trajectory_to_bev_ax(ax[1], human_trajectory, TRAJECTORY_CONFIG["human"])

    agent_trajectory = _agent_trajectory_for_frame(scene, absolute_predicted_poses, frame_idx)
    if agent_trajectory is not None:
        add_trajectory_to_bev_ax(ax[1], agent_trajectory, TRAJECTORY_CONFIG["agent"])

    for axis in ax:
        axis.set_xticks([])
        axis.set_yticks([])
    configure_bev_ax(ax[1])
    fig.tight_layout()
    fig.subplots_adjust(wspace=0.02, hspace=0.01, left=0.01, right=0.99, top=0.92, bottom=0.02)


def load_predictions(jsonl_path: Path) -> Dict[str, np.ndarray]:
    """Parse detailed_logs.jsonl into {token: predicted_poses[8,3]}."""
    preds: Dict[str, np.ndarray] = {}
    with open(jsonl_path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            token = obj.get("token")
            traj = obj.get("trajectory_se2")
            if token is None or traj is None:
                continue
            arr = np.asarray(traj, dtype=np.float32)
            if arr.shape == (TRAJECTORY_SAMPLING.num_poses, 3):
                preds[token] = arr
    return preds


def select_tokens(
    csv_path: Path,
    predictions: Dict[str, np.ndarray],
    token: Optional[str],
    topk_worst: Optional[int],
    metric: str,
    max_scenes: Optional[int],
) -> pd.DataFrame:
    """Pick tokens to render and return corresponding CSV rows."""
    df = pd.read_csv(csv_path)
    df = df[df["token"].isin(predictions.keys())].copy()

    if token is not None:
        sel = df[df["token"] == token]
        if sel.empty:
            raise SystemExit(f"Token {token} not found in {csv_path}")
        return sel

    if topk_worst is not None:
        if metric not in df.columns:
            raise SystemExit(f"metric '{metric}' not in CSV columns: {df.columns.tolist()}")
        sel = df.sort_values(metric, ascending=True).head(topk_worst)
        return sel

    # default: first max_scenes rows
    return df.head(max_scenes or 5)


def render_scene_gif(
    scene,
    predicted_poses: np.ndarray,
    title: str,
    out_path: Path,
    duration_ms: int,
) -> None:
    """Render one GIF: each frame = front camera + BEV (with two trajectories)."""
    num_history = scene.scene_metadata.num_history_frames
    current_idx = num_history - 1

    absolute_predicted_poses = _absolute_predicted_poses(scene, predicted_poses)

    images: List[Image.Image] = []
    for frame_idx in range(len(scene.frames)):
        fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
        _plot_front_and_bev(fig, ax, scene, frame_idx, absolute_predicted_poses)

        rel_t = (frame_idx - current_idx) * 0.5
        fig.suptitle(
            f"{title}  |  frame {frame_idx}/{len(scene.frames) - 1}  "
            f"(t={rel_t:+.1f}s)  green=GT  orange=pred",
            fontsize=10,
        )
        fig.tight_layout()
        fig.subplots_adjust(
            wspace=0.01, hspace=0.01, left=0.01, right=0.99, top=0.95, bottom=0.01
        )

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=90)
        buf.seek(0)
        images.append(Image.open(buf).copy())
        buf.close()
        plt.close(fig)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        out_path,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
    )


def render_snapshot(
    scene,
    predicted_poses: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """Render a single PNG snapshot at the current frame only (fast mode)."""
    num_history = scene.scene_metadata.num_history_frames
    current_idx = num_history - 1

    absolute_predicted_poses = _absolute_predicted_poses(scene, predicted_poses)
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.35, 1.0]})
    _plot_front_and_bev(fig, ax, scene, current_idx, absolute_predicted_poses)
    fig.suptitle(f"{title}  |  green=GT  orange=pred", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--exp_dir", type=Path, required=True,
                        help="Directory containing <timestamp>.csv and detailed_logs.jsonl")
    parser.add_argument("--data_root", type=Path, required=True,
                        help="OPENSCENE_DATA_ROOT, e.g. datasets/navsim_test")
    parser.add_argument("--split", type=str, default="test",
                        help="Data split folder under navsim_logs / sensor_blobs (default: test)")
    parser.add_argument("--csv", type=Path, default=None,
                        help="Override CSV path (default: auto-detect in exp_dir)")
    parser.add_argument("--jsonl", type=Path, default=None,
                        help="Override detailed_logs.jsonl path (default: <exp_dir>/detailed_logs.jsonl)")
    parser.add_argument("--output_dir", type=Path, default=None,
                        help="Output dir (default: <exp_dir>/viz)")
    parser.add_argument("--token", type=str, default=None, help="Render a single token")
    parser.add_argument("--topk_worst", type=int, default=None,
                        help="Render the N worst-scoring tokens by --metric")
    parser.add_argument("--metric", type=str, default="score",
                        help="Metric used for worst selection (default: score)")
    parser.add_argument("--max_scenes", type=int, default=5,
                        help="Default number of tokens when neither --token nor --topk_worst is given")
    parser.add_argument("--snapshot", action="store_true",
                        help="Only render a single PNG per token (fast, no GIF)")
    parser.add_argument("--duration_ms", type=int, default=300,
                        help="GIF frame duration in ms")
    args = parser.parse_args()

    exp_dir: Path = args.exp_dir.resolve()
    data_root: Path = args.data_root.resolve()

    csv_path = args.csv
    if csv_path is None:
        csv_candidates = sorted(exp_dir.glob("*.csv"))
        if not csv_candidates:
            raise SystemExit(f"no CSV found in {exp_dir}")
        csv_path = csv_candidates[0]
    jsonl_path = args.jsonl or (exp_dir / "detailed_logs.jsonl")
    output_dir = args.output_dir or (exp_dir / "viz")

    _bootstrap_env(data_root)

    print(f"[info] exp_dir    = {exp_dir}")
    print(f"[info] csv        = {csv_path}")
    print(f"[info] jsonl      = {jsonl_path}")
    print(f"[info] data_root  = {data_root} (split={args.split})")
    print(f"[info] output_dir = {output_dir}")

    predictions = load_predictions(jsonl_path)
    print(f"[info] predictions loaded: {len(predictions)} tokens")

    selected = select_tokens(
        csv_path=csv_path,
        predictions=predictions,
        token=args.token,
        topk_worst=args.topk_worst,
        metric=args.metric,
        max_scenes=args.max_scenes,
    )
    tokens = selected["token"].tolist()
    print(f"[info] rendering {len(tokens)} tokens")

    scene_filter = SceneFilter(
        num_history_frames=4,
        num_future_frames=10,
        frame_interval=1,
        has_route=True,
        tokens=tokens,
    )

    sensor_config = SensorConfig(
        cam_f0=True, cam_l0=False, cam_l1=False, cam_l2=False,
        cam_r0=False, cam_r1=False, cam_r2=False, cam_b0=False,
        lidar_pc=False,
    )
    scene_loader = SceneLoader(
        data_path=data_root / f"navsim_logs/{args.split}",
        original_sensor_path=data_root / f"sensor_blobs/{args.split}",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
    )

    available = set(scene_loader.tokens)
    missing = [t for t in tokens if t not in available]
    if missing:
        print(f"[warn] {len(missing)} tokens missing in logs (first 3: {missing[:3]})")

    score_col_map = {row["token"]: row for _, row in selected.iterrows()}

    for token in tqdm(tokens, desc="render"):
        if token not in available:
            continue
        try:
            scene = scene_loader.get_scene_from_token(token)
            row = score_col_map[token]
            title = (
                f"token={token}  score={row.get('score', float('nan')):.3f}  "
                f"prog={row.get('ego_progress', float('nan')):.2f}  "
                f"ttc={row.get('time_to_collision_within_bound', float('nan')):.2f}"
            )
            if args.snapshot:
                out_path = output_dir / f"{token}.png"
                render_snapshot(scene, predictions[token], title, out_path)
            else:
                out_path = output_dir / f"{token}.gif"
                render_scene_gif(
                    scene,
                    predictions[token],
                    title,
                    out_path,
                    duration_ms=args.duration_ms,
                )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(f"[error] failed {token}: {exc}")
            traceback.print_exc()

    print(f"[done] outputs in {output_dir}")


if __name__ == "__main__":
    main()
