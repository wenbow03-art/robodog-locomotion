cmake_minimum_required(VERSION 3.5)

project(robot_common_interface)

find_package(std_msgs REQUIRED)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

# set(srv_files
#   "srv/AddDiagnostics.srv"
#   "srv/SelfTest.srv"
# )

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/JointCommand.msg"
  "msg/ControlSource.msg"
  "msg/RobotException.msg"
  "msg/RobotActionData.msg"
  "msg/RobotControlFeedback.msg"
  "msg/MotionCommands.msg"
  "msg/TouchSensor.msg"
  "msg/JointActuatorData.msg"
  # ${srv_files}
  "srv/GaitSwitch.srv"
  "srv/SimulationReset.srv"
  "srv/RobotControlCommand.srv"
  DEPENDENCIES std_msgs ament_cmake geometry_msgs sensor_msgs
  ADD_LINTER_TESTS
)

ament_export_dependencies(rosidl_default_runtime)

ament_package()