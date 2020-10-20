#! /usr/bin/env python

from movement_primitives.srv import *
import rospy

def handle_promp_train(req):
    resp = PrompTrainResponse()
    std_msgs
    resp.traj_data.layout.dim = []
    resp.traj_data.layout.dim[0].label = "rows"
    resp.traj_data.layout.dim[0].size = 2
    resp.traj_data.layout.dim[0].stride = 0
    resp.traj_data.layout.dim[1].label = "cols"
    resp.traj_data.layout.dim[1].size = 3
    resp.traj_data.layout.dim[1].stride = 0
    resp.traj_data.layout.data_offset = 0
    resp.traj_data.data[0] = [4,3,5,3,2,4]
    return resp

def promp_train_server():
    rospy.init_node('promp_train_server')
    s = rospy.Service('promp_train', PrompTrain, handle_promp_train)
    print("Ready to ProMP Train.")
    rospy.spin()

if __name__ == "__main__":
    promp_train_server()
