#!/usr/bin/env bash

#./build_cpp.sh \
#    -DIMAGESTREAMIO_ROOT="$HOME/Documents/baldr/dcs/libImageStreamIO"

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build_dir="${script_dir}/build"

cmake -S "$script_dir" -B "$build_dir" "$@"
cmake --build "$build_dir" --parallel

printf '\nBuilt executables:\n'
printf '  %s\n' "$script_dir/shm_creator_sim" "$script_dir/sim_mdm_server"
