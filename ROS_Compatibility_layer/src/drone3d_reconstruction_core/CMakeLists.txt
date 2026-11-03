cmake_minimum_required(VERSION 3.0.2)
project(drone3d_reconstruction_core)

find_package(catkin REQUIRED COMPONENTS drone3d_runtime
  actionlib
  actionlib_msgs
  diagnostic_msgs
  drone3d_reconstruction_msgs
  nav_msgs
  rospy
  sensor_msgs
  std_srvs
  tf2_msgs
)

catkin_python_setup()
catkin_package(
  CATKIN_DEPENDS drone3d_runtime actionlib actionlib_msgs diagnostic_msgs drone3d_reconstruction_msgs nav_msgs rospy sensor_msgs std_srvs tf2_msgs
)

catkin_install_python(PROGRAMS
  scripts/reconstruct_production_node.py
  scripts/capture_production_node.py
  scripts/camera_pipeline_service_node.py
  scripts/continuous_session_node.py
  scripts/continuous_production_preflight.py
  scripts/continuous_web_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/continuous_production.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(PROGRAMS scripts/nx_step2_bringup.sh scripts/nx_record.sh scripts/camera_check.py scripts/nx_capture.sh
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(FILES docs/NX_BRINGUP.md docs/G4_INTEGRATION.md docs/ONBOARD_V1.md
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

if(CATKIN_ENABLE_TESTING)
  find_package(rostest REQUIRED)
  catkin_add_nosetests(src/g4_backend/tests)
  catkin_add_nosetests(src/recon_core/tests)
  catkin_add_nosetests(test/test_onboard.py)
  catkin_add_nosetests(test/test_g4_map_backend.py)
  catkin_add_nosetests(test/test_g4_input_adapter.py)
  catkin_add_nosetests(test/test_capture.py)
  catkin_add_nosetests(test/test_capture_action_server.py)
  catkin_add_nosetests(test/test_capture_node.py)
  catkin_add_nosetests(test/test_continuous_fault_matrix.py)
  catkin_add_nosetests(test/test_continuous_session.py)
  catkin_add_nosetests(test/test_continuous_session_node.py)
  catkin_add_nosetests(test/test_continuous_web.py)
  catkin_add_nosetests(test/test_continuous_web_node.py)
  catkin_add_nosetests(test/test_continuous_production_preflight.py)
  catkin_add_nosetests(test/test_cuda_preflight.py)
  catkin_add_nosetests(test/test_depth_metrics.py)
  catkin_add_nosetests(test/test_formal_result.py)
  catkin_add_nosetests(test/test_frame_pool.py)
  catkin_add_nosetests(test/test_job_controller.py)
  catkin_add_nosetests(test/test_odom_local.py)
  catkin_add_nosetests(test/test_shared_point_sim3.py)
  catkin_add_nosetests(test/test_shared_point_backend.py)
  catkin_add_nosetests(test/test_runtime_memory_guard.py)
  catkin_add_nosetests(test/test_pipeline_service.py)
  catkin_add_nosetests(test/test_pipeline_service_node.py)
  catkin_add_nosetests(test/test_production.py)
  catkin_add_nosetests(test/test_production_node.py)
  catkin_add_nosetests(test/test_reconstruct_action_server.py)
  catkin_add_nosetests(test/test_resource_gate.py)
  catkin_add_nosetests(test/test_result_transaction.py)
  catkin_add_nosetests(test/test_snapshot_validator.py)
  catkin_add_nosetests(test/test_storage_retention.py)
  catkin_add_nosetests(test/test_worker_runner.py)
  add_rostest(test/reconstruct_action_server.test)
  add_rostest(test/reconstruct_production_action_server.test)
endif()
