#! /usr/bin/env python
import rospy
from std_msgs.msg import String


def main():
	rospy.init_node('topic_publisher')
	pub = rospy.Publisher('phrases', String, queue_size=10)
	
	rate = rospy.Rate(2) #Setting a publish rate of 2 Hz
	msg_str = String()
	msg_str = 'Ground Control to Major Tom'

#	while not rospy.is_shutdown():
#		pub.publish(msg_str)
#		rate.sleep()
	pub.publish(msg_str)
	print(msg_str)
	rate.sleep()

if __name__=='__main__':
	main()
