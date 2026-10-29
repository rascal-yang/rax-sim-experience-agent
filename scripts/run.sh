#!/usr/bin/env bash
set -euo pipefail
TASK_REPO=$(cd "$(dirname "$0")/.." && pwd)
export PYTHONPATH="$TASK_REPO/src:$TASK_REPO/../cap-sim-api-augmentation:$TASK_REPO/../cap-sim-api-augmentation/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$TASK_REPO/../cap-sim/scripts/humano_python.sh" -m experience_agent.cli "$@"
