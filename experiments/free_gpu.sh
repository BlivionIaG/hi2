#!/usr/bin/env bash
# Print the first of our HIP devices (2-5) that is idle (0 % use, < 5 % VRAM, no KFD
# process, no *.gpuN.lock in the runs dir) on two checks 20 s apart; print nothing if none.
B=/home/chenco_adm/w4a8_runs/hadamard-int2
check() {
  for hip in 2 3 4 5; do
    smi=$((hip + 2))
    ls "$B"/*.gpu"$hip".lock >/dev/null 2>&1 && continue
    use=$(rocm-smi -d "$smi" --showuse | grep -o 'GPU use (%): [0-9]*' | grep -o '[0-9]*$')
    vram=$(rocm-smi -d "$smi" --showmemuse | grep -o 'VRAM%): [0-9]*' | grep -o '[0-9]*$')
    pids=$(rocm-smi --showpidgpus 2>/dev/null | grep -A1 '^PID' | grep -cx "$smi *")
    [ "${use:-1}" -eq 0 ] && [ "${vram:-99}" -lt 5 ] && [ "$pids" -eq 0 ] && echo "$hip"
  done
}
a=$(check); sleep 20; b=$(check)
for h in $a; do echo "$b" | grep -qx "$h" && { echo "$h"; exit 0; }; done
exit 0
