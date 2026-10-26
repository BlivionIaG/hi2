#!/usr/bin/env bash
# Run a command once one of our GPUs is really idle (other users do not take our lock files).
# usage: when_idle.sh "<hip indices>" <command...>     e.g. when_idle.sh "2 3 4 5" bash job.sh
# Idle = rocm-smi GPU use 0 % and VRAM under 5 % on three checks 20 s apart; on this box
# rocm-smi GPU[i] is HIP device i-2 (RSMI_OFFSET). The command runs with HIP_VISIBLE_DEVICES
# set to the chosen device. Check again right before anything that allocates a whole GPU.
set -u
read -ra CANDS <<< "$1"; shift
idle() {
  local r=$(($1 + ${RSMI_OFFSET:-2})) use mem
  use=$(rocm-smi -d "$r" --showuse | awk -F: '/GPU use/{gsub(/ /,"",$NF); print $NF}')
  mem=$(rocm-smi -d "$r" --showmemuse | awk -F: '/Allocated/{gsub(/ /,"",$NF); print $NF}')
  [ "${use:-100}" = 0 ] && [ "${mem:-100}" -lt 5 ]
}
while true; do
  for h in "${CANDS[@]}"; do
    ok=0
    while idle "$h"; do
      ok=$((ok + 1))
      if [ "$ok" -ge 3 ]; then HIP_VISIBLE_DEVICES=$h exec "$@"; fi
      sleep 20
    done
  done
  sleep 30
done
