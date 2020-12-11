Notes on ROS architecture:

1. Training and querying is implemented using ROS nodes (Publishers and Subscribers)
  Training:
2. tphsmm_train_caller.py takes the tphsmm_name and details about the demos, converts it to TPHSMMTrainMsg.msg and publishes it to topic: tphsmm_train
3. tphsmm_train_listener.py listens to topic: tphsmm_train for TPHSMMTrainMsg.msg, trains a TP-HSMM using this and saves the TP-HSMM to movement_primitives/saved_primitives
  Querying:
4. tphsmm_query_caller.py takes the tphsmm_name and via_pts, converts it to TPHSMMQueryTrigger.msg and publishes it to topic: tphsmm_query
5. tphsmm_query_listener.py listens to topic: tphsmm_query for TPHSMMQueryTrigger.msg, queries the required tphsmm and publishes the trajectory to RobotTraj as RobotTraj.msg



Running TP-HSMM training and querying using ROS:

In order to run the following, you need an active ros master (run roscore)

Training:
  Initializing the training listener:
  rosrun movement_primitives tphsmm_train_listener.py

  Training with a data set of demonstrations:
  rosrun movement_primitives tphsmm_train_caller.py '<name_of_promp_model>' '<path_to_dataset>' <num_demos> <num_frames>
  Example:
  rosrun movement_primitives tphsmm_train_caller.py 'tphsmm2' '/home/thejus/catkin_ws/src/movement_primitives/training_data/TP-HSMM_2frame' 10 2

Querying:
  Initializing the query listener:
  rosrun movement_primitives tphsmm_query_listener.py

  Querying with a set of via points:
  rosrun movement_primitives tphsmm_query_caller.py '<name_of_tphsmm_model>' <sensor_dofs> <each_dimension_of_via_pts_separated_by_commas>
  Example:
  rosrun movement_primitives tphsmm_query_caller.py 'tphsmm2' 2 500,1100,1500,1000
