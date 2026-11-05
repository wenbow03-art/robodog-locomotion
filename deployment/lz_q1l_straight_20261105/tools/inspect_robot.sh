#!/usr/bin/env bash
# Run on the robot after sourcing its ROS2 and SDK environment. Read-only.
set -uo pipefail
run() {
    printf '\n$'
    printf ' %q' "$@"
    printf '\n'
    timeout --signal=TERM --kill-after=2s 8s "$@"
    local rc=$?
    printf '[exit=%d]\n' "$rc"
}
run uname -a
run python3 --version
run python3 -c 'import torch; print("torch", torch.__version__)'
run systemctl show infer_zero.service --property=ActiveState,SubState,FragmentPath
printf '\nROS_DISTRO=%s ROS_DOMAIN_ID=%s RMW_IMPLEMENTATION=%s\n' "${ROS_DISTRO:-unset}" "${ROS_DOMAIN_ID:-unset}" "${RMW_IMPLEMENTATION:-unset}"
if ! command -v ros2 >/dev/null 2>&1; then
    printf 'ROS2 is not sourced. Source the robot ROS2 + SDK setup files and retry.\n' >&2
    exit 2
fi
run ros2 node list
run ros2 topic list -t
run ros2 interface show robot_common_interface/msg/JointCommand
run ros2 topic info /joint_actuator_commands --verbose
run ros2 topic info /joint_actuator_states --verbose
run ros2 topic info /imu --verbose
# Individual command status is retained in the log; this is inventory, not a readiness gate.
