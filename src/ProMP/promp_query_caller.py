#!/usr/bin/env python
import rospy
from std_msgs.msg import String
from movement_primitives.msg import *

def promp_query_caller(promp_name, input_dofs, input):
    pub = rospy.Publisher('promp_query', PrompQueryTrigger, queue_size=10)
    rospy.init_node('promp_query_caller', anonymous=True)
    rate = rospy.Rate(10) # 10hz
    # while not rospy.is_shutdown():
    print(type(input_dofs))
    msg = PrompQueryTrigger()
    msg.promp_name = promp_name
    msg.input_dofs = input_dofs
    msg.input = input
    rospy.loginfo(msg)
    pub.publish(msg)
    # rate.sleep()

if __name__ == '__main__':
    try:
        if len(sys.argv) == 4:
            promp_name = str(sys.argv[1])
            input_dofs = int(sys.argv[2])
            input_str = list(sys.argv[3].split(','))
            input = [int(i) for i in input_str]
            promp_query_caller(promp_name, input_dofs, input)
        else:
            print('Error: More arguments required')
    except rospy.ROSInterruptException:
        pass
