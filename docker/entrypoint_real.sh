#!/usr/bin/env bash
set -e

source /opt/ros/humble/setup.bash

if [[ -f /px4_msgs_ws/install/setup.bash ]]; then
  source /px4_msgs_ws/install/setup.bash
fi

if [[ -f /zed_ws/install/setup.bash ]]; then
  source /zed_ws/install/setup.bash
fi

if [[ -f /ws/install/setup.bash ]]; then
  source /ws/install/setup.bash
fi

exec "$@"
