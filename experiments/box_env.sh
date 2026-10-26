#!/usr/bin/env bash
# Source on the gfx1030 box: the venv, its ROCm libraries and PYTHONPATH for hi2.
# usage: . experiments/box_env.sh   (from the hi2 copy, e.g. hi2-fast/)
V=/home/chenco_adm/Apps/vllm/venv-7.14.0_0.31.0-final
SL=$V/lib/python3.12/site-packages
export LD_LIBRARY_PATH=$SL/_rocm_sdk_libraries/lib:$SL/_rocm_sdk_core/lib/host-math/lib:$SL/_rocm_sdk_core/lib/rocm_sysdeps/lib:$SL/_rocm_sdk_core/lib/core/lib:$SL/torch/lib
HI2_HOME=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$HI2_HOME
export PY=$V/bin/python
