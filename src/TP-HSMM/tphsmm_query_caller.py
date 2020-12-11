#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from movement_primitives.msg import *

def tphsmm_query_caller(tphsmm_name, input_dofs, input):
    pub = rospy.Publisher('tphsmm_query', TPHSMMQueryTrigger, queue_size=10)
    rospy.init_node('tphsmm_query_caller', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    print(type(input_dofs))
    msg = TPHSMMQueryTrigger()
    msg.tphsmm_name = tphsmm_name
    msg.input_dofs = input_dofs
    msg.input = input
    print("Message Generated")
    rospy.loginfo(msg)
    pub.publish(msg)
    # rate.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) == 4:
            tphsmm_name = str(sys.argv[1])
            input_dofs = int(sys.argv[2])
            input_str = list(sys.argv[3].split(','))
            input = [int(i) for i in input_str]
            tphsmm_query_caller(tphsmm_name, input_dofs, input)
        else:
            print('Error: More arguments required')
    except rospy.ROSInterruptException:
        pass
