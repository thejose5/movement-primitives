Notes on ROS architecture:

1. Training and querying is implemented using ROS nodes (Publishers and Subscribers)
  Training:
2. promp_train_caller.py takes the promp_name and details about the demos, converts it to PrompTrainMsg.msg and publishes it to topic: promp_train
3. promp_train_listener.py listens to topic: promp_train for PrompTrainMsg.msg, trains a ProMP using this and saves the ProMP to movement_primitives/saved_primitives
  Querying:
4. promp_query_caller.py takes the promp_name and via_pts, converts it to PrompQueryTrigger.msg and publishes it to topic: promp_query
5. promp_query_listener.py listens to topic: promp_query for PrompQueryTrigger.msg, queries the required promp and publishes the trajectory to RobotTraj as RobotTraj.msg



Running ProMP training and querying using ROS:

In order to run the following, you need an active ros master (run roscore)

Training:
  Initializing the training listener:
  rosrun movement_primitives promp_train_listener.py

  Training with a data set of demonstrations:
  rosrun movement_primitives promp_train_caller.py '<name_of_promp_model>' '<path_to_dataset' <sensor_dofs> <robot_dofs>
  Example:
  rosrun movement_primitives promp_train_caller.py 'pmp1' '/home/thejus/catkin_ws/src/movement_primitives/training_data/ProMP' 4 3

Querying:
  Initializing the query listener:
  rosrun movement_primitives promp_query_listener.py

  Querying with a set of via points:
  rosrun movement_primitives promp_query_caller.py '<name_of_promp_model>' <sensor_dofs> <each_dimension_of_via_pts_separated_by_commas>
  Example:
  rosrun movement_primitives promp_query_caller.py 'pmp1' 4 500,1000,1500,1000
