#!/usr/bin/env zsh
set -euo pipefail
mkdir -p results
make -B
tail -n +1 results/ptxas.txt | sed -n '1,8p'
