#!/usr/bin/env python

from __future__ import print_function

import sys
import rospy
from movement_primitives.srv import *

def promp_train_client():
    rospy.wait_for_service('promp_train')
    try:
        promp_train = rospy.ServiceProxy('promp_train', PrompTrain)
        promp_name = "abc"
        resp1 = promp_train(promp_name)
        return resp1
    except rospy.ServiceException as e:
        print("Service call failed: %s"%e)

def usage():
    return "%s [x y]"%sys.argv[0]

if __name__ == "__main__":
    resp = promp_train_client()
    print(resp.step)
    print(resp.traj_data)
