#!/bin/bash
pkill -9 gunicorn

export NUPLAN_MAP_VERSION="nuplan-maps-v1.0"
export NAVSIM_DEVKIT_ROOT="ABS_PATH_TO_NAVSIM_EVAL"
export OPENSCENE_DATA_ROOT="ABS_PATH_TO_OPENSCENE_DATA"
export NAVSIM_EXP_ROOT="EXP_ROOT_PATH"
export NUPLAN_MAPS_ROOT="ABS_PATH_TO_NUPLAN_MAPS"
export CACHE_PATH=$NAVSIM_EXP_ROOT/metric_cache_train
HOST="0.0.0.0"
PORT=8901

NUM_WORKERS=192

echo "[INFO] Starting Gunicorn server on $HOST:$PORT with $NUM_WORKERS workers..."

gunicorn navsim.planning.script.run_gunicorn_server:app \
    -w $NUM_WORKERS \
    -k uvicorn.workers.UvicornWorker \
    -b $HOST:$PORT \
    --timeout 150 \
    --log-level info \

echo "[INFO] Gunicorn server started."