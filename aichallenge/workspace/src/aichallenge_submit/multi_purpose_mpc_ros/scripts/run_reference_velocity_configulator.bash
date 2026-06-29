#!/bin/bash
# shellcheck disable=SC1091
VENV_ACTIVATE="$(ros2 pkg prefix multi_purpose_mpc_ros)/.venv/bin/activate"
if [ -f "$VENV_ACTIVATE" ]; then
    source "$VENV_ACTIVATE"
fi
python3 "$(ros2 pkg prefix multi_purpose_mpc_ros)/lib/multi_purpose_mpc_ros/reference_velocity_configulator" "$@"
