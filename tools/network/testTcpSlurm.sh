#!/usr/bin/env bash
set -Eeuo pipefail
export FOAMNORDIC_NETWORK_TRANSPORT=tcp
exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/testNetworkSlurm.sh" "$@"
