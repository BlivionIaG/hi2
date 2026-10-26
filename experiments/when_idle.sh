#!/usr/bin/env bash
# Run a command once a GPU is really idle (other users do not take our lock files).
# usage: when_idle.sh <hip index> <command...>
# Idle = rocm-smi GPU use 0 % and VRAM under 5 % on three checks 20 s apart; on this box
# rocm-smi GPU[i] is HIP device i-2 (RSMI_OFFSET). The command runs with HIP_VISIBLE_DEVICES set.
set -u
HIP=$1; shift
R=$((HIP + ${RSMI_OFFSET:-2}))
ok=0
while [ "$ok" -lt 3 ]; do
  use=$(rocm-smi -d "$R" --showuse | awk -F: '/GPU use/{gsub(/ /,"",$NF); print $NF}')
  mem=$(rocm-smi -d "$R" --showmemuse | awk -F: '/Allocated/{gsub(/ /,"",$NF); print $NF}')
  if [ "${use:-100}" = 0 ] && [ "${mem:-100}" -lt 5 ]; then ok=$((ok + 1)); else ok=0; fi
  sleep 20
done
HIP_VISIBLE_DEVICES=$HIP exec "$@"
