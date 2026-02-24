#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [[ -f /px4_msgs_ws/install/setup.bash ]]; then
  source /px4_msgs_ws/install/setup.bash
fi

if [[ -f /ws/install/setup.bash ]]; then
  source /ws/install/setup.bash
fi

worlds_dir=/ws/src/ai_scanner/simulation/gazebo/worlds
if [[ -d "$worlds_dir" ]]; then
  export IGN_GAZEBO_RESOURCE_PATH="$worlds_dir${IGN_GAZEBO_RESOURCE_PATH:+:$IGN_GAZEBO_RESOURCE_PATH}"
  export GZ_SIM_RESOURCE_PATH="$worlds_dir${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
fi

exec "$@"