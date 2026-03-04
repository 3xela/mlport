#!/usr/bin/env zsh
set -euo pipefail
mkdir -p results
ncu --set default --export results/ncu.ncu-rep ./main --n ${1:-67108864} --block ${2:-256}
