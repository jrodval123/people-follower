#! /usr/bin/env python
import rospy
from std_msgs.msg import String

def callback(msg):
	print msg.data

def main():
	rospy.init_node('topic_sucriber')
	sub = rospy.Subscriber('/detector', String, callback)
	rospy.spin()

if __name__=='__main__':
	main()
