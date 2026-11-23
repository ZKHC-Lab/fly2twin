cmake_minimum_required(VERSION 3.0.2)
project(drone3d_reconstruction_msgs)

find_package(catkin REQUIRED COMPONENTS
  actionlib_msgs
  message_generation
  std_msgs
  sensor_msgs
)

add_action_files(
  DIRECTORY action
  FILES
    CaptureSegment.action
    ReconstructSegment.action
)

add_service_files(
  DIRECTORY srv
  FILES
    CaptureControl.srv
    ContinuousSessionCommand.srv
)

generate_messages(DEPENDENCIES actionlib_msgs std_msgs sensor_msgs)

catkin_package(CATKIN_DEPENDS actionlib_msgs message_runtime std_msgs sensor_msgs)

if(CATKIN_ENABLE_TESTING)
  catkin_add_nosetests(test/test_message_contract.py)
endif()
