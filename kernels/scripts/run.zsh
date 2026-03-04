#!/usr/bin/env zsh
set -euo pipefail
./main --n ${1:-67108864} --block ${2:-256} --warmup ${3:-100} --iters ${4:-1000}
