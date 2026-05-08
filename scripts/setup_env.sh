#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
LLAMAFACTORY_ROOT="$PROJECT_ROOT/LLaMA-Factory"
LLAMAFACTORY_REPO_URL="https://github.com/hiyouga/LLaMA-Factory.git"
UV_ENV_ROOT="$PROJECT_ROOT/.venvs"
CURIOUS_ENV="$UV_ENV_ROOT/curious"
NAVSIM_ENV="$UV_ENV_ROOT/navsim"
PIP_INDEX_URL="http://mirrors.tencentyun.com/pypi/simple"
CURIOUS_PYTHON="3.10"
NAVSIM_PYTHON="3.9"
LLAMAFACTORY_VERSION="0.9.3"
GIT_CLONE_PROXY="http://127.0.0.1:1087"
NUPLAN_DEVKIT_ROOT="$PROJECT_ROOT/third_party/nuplan-devkit"
NUPLAN_DEVKIT_REF="nuplan-devkit-v1.2"
NUPLAN_DEVKIT_REPO_URL="https://github.com/motional/nuplan-devkit.git"

export no_proxy=tencentyun.com
export UV_INDEX_URL="$PIP_INDEX_URL"
if [[ "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  bash scripts/setup_env.sh

This script:
  - creates `.venvs/curious` (Python 3.10) with uv
  - installs EasyR1 into `.venvs/curious`
  - clones `LLaMA-Factory/` if missing (proxy only for clone)
  - installs `llamafactory==0.9.3` into `.venvs/curious` (Python 3.10 compatible)
  - creates `.venvs/navsim` (Python 3.9) with uv
  - clones `nuplan-devkit` if missing (proxy only for clone)
  - installs navsim requirements and `navsim_eval` into `.venvs/navsim`
  - uses mirror: http://mirrors.tencentyun.com/pypi/simple

Data setup is still manual. See:
  - `navsim_eval/docs/install.md`
  - `docs/deploy.md`
  - `docs/train_sft.md`
EOF
    exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is required but not found in PATH." >&2
    exit 1
fi

# Keep network tools unproxied by default.
# unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY all_proxy ALL_PROXY no_proxy NO_PROXY

mkdir -p "$UV_ENV_ROOT"

echo "Setting up curious (.venvs/curious)..."
if [[ -x "$CURIOUS_ENV/bin/python" ]]; then
    CURIOUS_CURRENT="$("$CURIOUS_ENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$CURIOUS_CURRENT" != "$CURIOUS_PYTHON" ]]; then
        echo "Recreating curious env: need Python $CURIOUS_PYTHON, found $CURIOUS_CURRENT"
        rm -rf "$CURIOUS_ENV"
    fi
fi
if [[ ! -x "$CURIOUS_ENV/bin/python" ]]; then
    uv venv --python "$CURIOUS_PYTHON" "$CURIOUS_ENV"
fi

uv pip install --python "$CURIOUS_ENV/bin/python" --index-url "$PIP_INDEX_URL" -e "$PROJECT_ROOT/EasyR1"

if [[ ! -d "$LLAMAFACTORY_ROOT/.git" ]]; then
    if [[ -e "$LLAMAFACTORY_ROOT" ]]; then
        echo "Cannot clone LLaMA-Factory into: $LLAMAFACTORY_ROOT" >&2
        exit 1
    fi
    https_proxy="$GIT_CLONE_PROXY" HTTPS_PROXY="$GIT_CLONE_PROXY" \
        git clone --depth 1 "$LLAMAFACTORY_REPO_URL" "$LLAMAFACTORY_ROOT"
fi

uv pip install --python "$CURIOUS_ENV/bin/python" --index-url "$PIP_INDEX_URL" "llamafactory==${LLAMAFACTORY_VERSION}"

echo "Setting up navsim (.venvs/navsim)..."
if [[ -x "$NAVSIM_ENV/bin/python" ]]; then
    NAVSIM_CURRENT="$("$NAVSIM_ENV/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$NAVSIM_CURRENT" != "$NAVSIM_PYTHON" ]]; then
        echo "Recreating navsim env: need Python $NAVSIM_PYTHON, found $NAVSIM_CURRENT"
        rm -rf "$NAVSIM_ENV"
    fi
fi
if [[ ! -x "$NAVSIM_ENV/bin/python" ]]; then
    uv venv --python "$NAVSIM_PYTHON" "$NAVSIM_ENV"
fi

if [[ ! -d "$NUPLAN_DEVKIT_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$NUPLAN_DEVKIT_ROOT")"
    https_proxy="$GIT_CLONE_PROXY" HTTPS_PROXY="$GIT_CLONE_PROXY" \
        git clone --depth 1 --branch "$NUPLAN_DEVKIT_REF" "$NUPLAN_DEVKIT_REPO_URL" "$NUPLAN_DEVKIT_ROOT"
fi

uv pip install --python "$NAVSIM_ENV/bin/python" --index-url "$PIP_INDEX_URL" "$NUPLAN_DEVKIT_ROOT"
NAVSIM_REQ_NO_NUPLAN="$(mktemp)"
awk '!/^nuplan-devkit @ git\+/' "$PROJECT_ROOT/navsim_eval/requirements.txt" > "$NAVSIM_REQ_NO_NUPLAN"
uv pip install --python "$NAVSIM_ENV/bin/python" --index-url "$PIP_INDEX_URL" -r "$NAVSIM_REQ_NO_NUPLAN"
rm -f "$NAVSIM_REQ_NO_NUPLAN"
uv pip install --python "$NAVSIM_ENV/bin/python" --index-url "$PIP_INDEX_URL" --no-deps -e "$PROJECT_ROOT/navsim_eval"
"$NAVSIM_ENV/bin/python" -c "import torch" >/dev/null 2>&1 || \
    uv pip install --python "$NAVSIM_ENV/bin/python" --index-url "$PIP_INDEX_URL" torch

echo "Setup complete."
echo "curious python: $CURIOUS_ENV/bin/python"
echo "navsim  python: $NAVSIM_ENV/bin/python"
echo "Optional: uv pip install --python $CURIOUS_ENV/bin/python --index-url $PIP_INDEX_URL flash-attn --no-build-isolation"
