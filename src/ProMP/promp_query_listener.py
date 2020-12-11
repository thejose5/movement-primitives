#!/usr/bin/env python
import rospy
import pickle
import numpy as np
from std_msgs.msg import String
from movement_primitives.msg import *
from inter_promp_train_py2 import *

rospy.init_node('promp_query_listener', anonymous=True)
traj_pub = rospy.Publisher('RobotTraj', RobotTraj, queue_size=10)

def callback(data):
    global traj_pub
    print(data.promp_name, data.input_dofs, data.input)
    PROMP_SAVE_PATH = "/home/thejus/catkin_ws/src/movement_primitives/saved_primitives/ProMP/" + data.promp_name
    pmpfile = open(PROMP_SAVE_PATH, 'rb')
    pmp = pickle.load(pmpfile)
    pmpfile.close()
    rospy.loginfo("ProMP Loaded")
    input_traj = trajectorizePoints(data.input, pmp)
    output_traj = pmp.predict(input_traj)
    rospy.loginfo("ProMP Queried")

    msg = RobotTraj()
    msg.robot_dofs = output_traj.shape[1]
    msg.traj = output_traj.reshape((output_traj.shape[0]*output_traj.shape[1],))
    traj_pub.publish(msg)
    plot_traj(output_traj, data.input)
    promp_query_listener()

def plot_traj(traj, input):
    plt.figure(1)
    plt.plot(traj[:,4],traj[:,5])
    plt.plot(input[0],input[1],'ro')
    plt.plot(input[2], input[3], 'ro')
    plt.show()

def promp_query_listener():
    rospy.Subscriber("promp_query", PrompQueryTrigger, callback)
    print('Ready for call to query ProMP')
    rospy.spin()

if __name__ == '__main__':
    promp_query_listener()
