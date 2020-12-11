#!/usr/bin/env python
import rospy
import pickle
import numpy as np
from std_msgs.msg import String
from movement_primitives.msg import *
from tphsmm_v2 import *

rospy.init_node('tphsmm_query_listener', anonymous=True)
traj_pub = rospy.Publisher('RobotTraj', RobotTraj, queue_size=10)

def callback(data):
    global traj_pub
    print(data.tphsmm_name, data.input_dofs, data.input)
    TPHSMM_SAVE_PATH = "/home/thejus/catkin_ws/src/movement_primitives/saved_primitives/TP-HSMM/" + data.tphsmm_name
    tphsmmfile = open(TPHSMM_SAVE_PATH, 'rb')
    tphsmm = pickle.load(tphsmmfile)
    tphsmmfile.close()
    rospy.loginfo("TP-HSMM Loaded")

    assert len(data.input) % data.input_dofs == 0
    query = [[]]
    for i in range(1,len(data.input)+1):
        query[-1].append(data.input[i-1])
        if i%data.input_dofs == 0 and i != len(data.input):
            query.append([])
    print(query)
    output_traj = tphsmm.predict(query)

    msg = RobotTraj()
    msg.robot_dofs = output_traj.shape[1]
    msg.traj = output_traj.reshape((output_traj.shape[0]*output_traj.shape[1],))
    traj_pub.publish(msg)
    plotTraj(output_traj, query)
    tphsmm_query_listener()


def tphsmm_query_listener():
    rospy.Subscriber("tphsmm_query", TPHSMMQueryTrigger, callback)
    print('Ready for call to query TP-HSMM.')
    rospy.spin()

if __name__ == '__main__':
    tphsmm_query_listener()
