#!/usr/bin/env python
import rospy
import pickle
from std_msgs.msg import String
from movement_primitives.msg import *
from tphsmm_v2 import *

def callback(data):
    rospy.loginfo("I heard %s, %s, %d, %d", data.tphsmm_name, data.path_to_data, data.num_demos, data.num_frames)
    data_addr = data.path_to_data
    ndemos = data.num_demos #len(os.listdir(data_addr))
    data_addr = data_addr+"/" #ProMP() requires that there be a '/' after the name of the target folder
    tphsmm = TP_HSMM(data_addr=data_addr, num_frames=data.num_frames, ndemos=ndemos)
    rospy.loginfo("TP-HSMM Trained. Saving...")

    TPHSMM_SAVE_PATH = "/home/thejus/catkin_ws/src/movement_primitives/saved_primitives/TP-HSMM/" + data.tphsmm_name
    outfile = open(TPHSMM_SAVE_PATH, 'wb')
    pickle.dump(tphsmm,outfile)
    outfile.close()
    rospy.loginfo("TP-HSMM saved in movement_primitives/saved_primitives/TP-HSMM under the name %s", data.tphsmm_name)
    tphsmm_train_listener()

def tphsmm_train_listener():
    rospy.init_node('tphsmm_listener', anonymous=True)
    rospy.Subscriber('tphsmm_train', TPHSMMTrainMsg, callback)
    print('Ready for call to train ProMP')
    rospy.spin()

if __name__ == '__main__':
    tphsmm_train_listener()
