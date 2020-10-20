#!/usr/bin/env python
import rospy
import pickle
from std_msgs.msg import String
from movement_primitives.msg import *
from inter_promp_train import *

def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s, %s, %d, %d", data.promp_name, data.path_to_data, data.sensor_dofs, data.robot_dofs)
    data_addr = data.path_to_data
    ndemos = len(os.listdir(data_addr))
    data_addr = data_addr+"/" #ProMP() requires that there be a '/' after the name of the target folder
    pmp = ProMP(ndemos=ndemos, hum_dofs=data.sensor_dofs, robot_dofs=data.robot_dofs, training_address=data_addr)
    rospy.loginfo("ProMP Trained. Saving...")
    PROMP_SAVE_PATH = "/home/thejus/catkin_ws/src/movement_primitives/saved_primitives/ProMP/" + data.promp_name
    outfile = open(PROMP_SAVE_PATH, 'wb')
    pickle.dump(pmp,outfile)
    outfile.close()
    rospy.loginfo("ProMP saved in movement_primitives/saved_primitives/ProMP under the name %s", data.promp_name)


def promp_train_listener():
    rospy.init_node('promp_train_listener', anonymous=True)
    rospy.Subscriber("promp_train", PrompTrainMsg, callback)
    print('Ready for call to train ProMP')
    rospy.spin()

if __name__ == '__main__':
    promp_train_listener()
