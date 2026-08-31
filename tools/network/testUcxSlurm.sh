#!/usr/bin/env bash
set -Eeuo pipefail
export FOAMNORDIC_NETWORK_TRANSPORT=ucx
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testNetworkSlurm.sh" "$@"
