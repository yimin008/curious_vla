#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
NAVSIM_DEVKIT_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
echo $PROJECT_ROOT
# Edit before running.
DATA_ROOT="$PROJECT_ROOT/datasets/navsim_test"
TRAIN_TEST_SPLIT="navtest"
CACHE_PATH="$PROJECT_ROOT/exp_root/metric_cache_${TRAIN_TEST_SPLIT}"
WORKER="${WORKER:-single_machine_thread_pool}"
USE_PROCESS_POOL="${USE_PROCESS_POOL:-True}"

# Ray writes temp/spilling files heavily; keep them off small /tmp.
if [[ -z "${RAY_TMP_BASE:-}" ]]; then
    if [[ -d "/data/dat" ]]; then
        RAY_TMP_BASE="/data/dat"
    elif [[ -d "/data/data" ]]; then
        RAY_TMP_BASE="/data/data"
    else
        RAY_TMP_BASE="/tmp"
    fi
fi
RAY_TMP_ROOT="${RAY_TMP_ROOT:-$RAY_TMP_BASE/ray_tmp}"
RAY_SPILL_ROOT="${RAY_SPILL_ROOT:-$RAY_TMP_BASE/ray_spill}"

if [[ ! -d "$DATA_ROOT" ]]; then
    echo "DATA_ROOT not found: $DATA_ROOT" >&2
    exit 1
fi

mkdir -p "$CACHE_PATH" "$RAY_TMP_ROOT" "$RAY_SPILL_ROOT"
echo "RAY_TMP_ROOT=$RAY_TMP_ROOT"
echo "RAY_SPILL_ROOT=$RAY_SPILL_ROOT"

source "$PROJECT_ROOT/.venvs/navsim/bin/activate"

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="$NAVSIM_DEVKIT_ROOT"
export OPENSCENE_DATA_ROOT="$DATA_ROOT"
export NAVSIM_EXP_ROOT="$PROJECT_ROOT/exp_root"
export NUPLAN_MAPS_ROOT="$DATA_ROOT/maps"
export TMPDIR="${TMPDIR:-$RAY_TMP_ROOT}"
export RAY_TMPDIR="$RAY_TMP_ROOT"
export RAY_object_spilling_config="{\"type\":\"filesystem\",\"params\":{\"directory_path\":\"$RAY_SPILL_ROOT\"}}"

uv run python "$NAVSIM_DEVKIT_ROOT/navsim/planning/script/run_metric_caching.py" \
    train_test_split="$TRAIN_TEST_SPLIT" \
    metric_cache_path="$CACHE_PATH" \
    worker="$WORKER" \
    worker.use_process_pool="$USE_PROCESS_POOL"
