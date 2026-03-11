# run_gunicorn_server_cached.py
# A/B test server that can cache MetricCache in RAM without modifying existing files.

import logging
import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional

import hydra
import numpy as np
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from navsim.common.dataloader import MetricCacheLoader
from navsim.planning.metric_caching.metric_cache import MetricCache
from nuplan.planning.script.builders.logging_builder import build_logger

from .helper_cached import score_same_token_trajectory_cached, score_single_trajectory_cached


hydra.initialize(config_path="config/pdm_scoring", version_base=None)
cfg = hydra.compose(config_name="default_run_pdm_score_server")
cfg.experiment_name = "gunicorn_server_cached"

build_logger(cfg)
logger = logging.getLogger(__name__)


class ScoreRequest(BaseModel):
    token: str
    poses: List[List[float]]
    verbose: Optional[bool] = False


class ScoreGroupRequest(BaseModel):
    token: str
    poses: List[List[List[float]]]
    verbose: Optional[bool] = False


# ------------------------------
# Cache strategy
# ------------------------------
# NAVSIM_CACHE_MODE:
#   - disk        : always load from disk (baseline)
#   - lru         : per-worker LRU cache (no code changes, safest)
#   - preload_all : preload ALL MetricCache objects into a dict
#
# NAVSIM_PRELOAD_AT_IMPORT=1 + gunicorn --preload:
#   Preloading happens in master before fork, enabling copy-on-write sharing across workers.
#   This is the only practical way to "share one in-memory copy" across gunicorn workers in Python.
#
_CACHE_MODE = os.getenv("NAVSIM_CACHE_MODE", "disk").lower()
_LRU_SIZE = int(os.getenv("NAVSIM_LRU_SIZE", "8192"))
_PRELOAD_LIMIT = int(os.getenv("NAVSIM_PRELOAD_LIMIT", "0"))  # 0 = no limit

_metric_cache_loader: Optional[MetricCacheLoader] = None
_metric_cache_dict: Optional[Dict[str, MetricCache]] = None
_get_metric_cache: Optional[Callable[[str], MetricCache]] = None


def _build_cache_objects() -> None:
    global _metric_cache_loader, _metric_cache_dict, _get_metric_cache

    cache_path = os.getenv("CACHE_PATH", cfg.metric_cache_path)
    cfg.metric_cache_path = cache_path
    loader = MetricCacheLoader(Path(cache_path))

    if _CACHE_MODE == "preload_all":
        logger.warning("Preloading ALL metric caches into RAM. This can take a long time.")
        metric_cache_dict: Dict[str, MetricCache] = {}
        tokens = loader.tokens
        if _PRELOAD_LIMIT > 0:
            tokens = tokens[:_PRELOAD_LIMIT]

        for i, token in enumerate(tokens, start=1):
            metric_cache_dict[token] = loader.get_from_token(token)
            if i % 1000 == 0:
                logger.info(f"preload progress: {i}/{len(tokens)}")

        _metric_cache_loader = loader
        _metric_cache_dict = metric_cache_dict
        _get_metric_cache = metric_cache_dict.__getitem__
        logger.warning(f"Preload done: {len(metric_cache_dict)} tokens")
        return

    if _CACHE_MODE == "lru":
        _metric_cache_loader = loader

        @lru_cache(maxsize=_LRU_SIZE)
        def _cached_load(token: str) -> MetricCache:
            return loader.get_from_token(token)

        _get_metric_cache = _cached_load
        logger.info(f"Using per-worker LRU cache: size={_LRU_SIZE}")
        return

    # default: disk
    _metric_cache_loader = loader
    _get_metric_cache = loader.get_from_token
    logger.info("Using disk cache loader (no in-RAM caching)")


# If you start with: gunicorn --preload ... and set NAVSIM_PRELOAD_AT_IMPORT=1,
# then this will run in the master process BEFORE workers fork.
if os.getenv("NAVSIM_PRELOAD_AT_IMPORT", "0") == "1":
    _build_cache_objects()


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: initializing cache loader...")
    app.state.cfg = cfg

    # If not preloaded at import time, build in each worker.
    if _get_metric_cache is None:
        _build_cache_objects()

    app.state.get_metric_cache = _get_metric_cache
    app.state.metric_cache_dict = _metric_cache_dict
    app.state.cache_mode = _CACHE_MODE
    logger.info(f"cache_mode={_CACHE_MODE}, metric_cache_path={cfg.metric_cache_path}")

    yield

    logger.info("Application shutdown.")


app = FastAPI(lifespan=lifespan)


@app.get("/ping")
def ping():
    return {"status": "ok", "cache_mode": app.state.cache_mode}


@app.get("/cache_stats")
def cache_stats():
    d = app.state.metric_cache_dict
    return {
        "cache_mode": app.state.cache_mode,
        "preloaded_tokens": 0 if d is None else len(d),
        "lru_size": _LRU_SIZE if app.state.cache_mode == "lru" else None,
    }


@app.post("/score")
def score_endpoint(req: ScoreRequest):
    get_metric_cache = app.state.get_metric_cache
    verbose = bool(req.verbose)

    metrics = score_single_trajectory_cached(
        app.state.cfg,
        get_metric_cache,
        req.token,
        req.poses,
        verbose=verbose,
    )

    for k in list(metrics.keys()):
        if isinstance(metrics[k], np.ndarray):
            metrics[k] = metrics[k].tolist()

    return metrics


@app.post("/score_group")
def score_group_endpoint(req: ScoreGroupRequest):
    get_metric_cache = app.state.get_metric_cache
    verbose = bool(req.verbose)

    metric_cache = get_metric_cache(req.token)
    metrics_list = score_same_token_trajectory_cached(app.state.cfg, metric_cache, req.poses, verbose=verbose)

    for metrics in metrics_list:
        for k in list(metrics.keys()):
            if isinstance(metrics[k], np.ndarray):
                metrics[k] = metrics[k].tolist()

    return metrics_list


def main():
    port = cfg.get("server", {}).get("port", 8902)
    uvicorn.run(app, host="0.0.0.0", port=port)


if __name__ == "__main__":
    main()
