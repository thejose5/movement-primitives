#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from movement_primitives.msg import *

def promp_train_caller(promp_name, path_to_data, sensor_dofs, robot_dofs):
    pub = rospy.Publisher('promp_train', PrompTrainMsg, queue_size=10)
    rospy.init_node('promp_train', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    msg = PrompTrainMsg()
    msg.promp_name = promp_name
    msg.path_to_data = path_to_data
    msg.sensor_dofs = sensor_dofs
    msg.robot_dofs = robot_dofs
    rospy.loginfo(msg)
    pub.publish(msg)
    # rate.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) == 5:
            promp_name = str(sys.argv[1])
            path_to_data = str(sys.argv[2])
            sensor_dofs = int(sys.argv[3])
            robot_dofs = int(sys.argv[4])
            promp_train_caller(promp_name, path_to_data, sensor_dofs, robot_dofs)
        else:
            print('Error: More arguments required')
    except rospy.ROSInterruptException:
        pass
