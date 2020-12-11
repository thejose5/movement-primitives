#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from movement_primitives.msg import *

def tphsmm_train_caller(promp_name, path_to_data, num_demos, num_frames):
    pub = rospy.Publisher('tphsmm_train', TPHSMMTrainMsg, queue_size=10)
    rospy.init_node('tphsmm_train', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    msg = TPHSMMTrainMsg()
    msg.tphsmm_name = promp_name
    msg.path_to_data = path_to_data
    msg.num_demos = num_demos
    msg.num_frames = num_frames
    rospy.loginfo(msg)
    pub.publish(msg)
    # rate.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) == 5:
            promp_name = str(sys.argv[1])
            path_to_data = str(sys.argv[2])
            num_demos = int(sys.argv[3])
            num_frames = int(sys.argv[4])
            tphsmm_train_caller(promp_name, path_to_data, num_demos, num_frames)
        else:
            print('Error: More arguments required')
    except rospy.ROSInterruptException:
        pass
