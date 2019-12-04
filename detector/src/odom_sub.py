#! /usr/bin/env python

import rospy
from nav_msgs.msg import Odometry


def callback(msg):
	x = msg.pose.pose.position.x
	y = msg.pose.pose.position.y
	z = msg.pose.pose.position.z
	print("X: {0} 	Y: {1}	Z: {2}	\n".format(round(x,2),round(y,2),round(z,2)))

def main():
	rospy.init_node('Odom_sub')
	odomSub = rospy.Subscriber('/odom', Odometry, callback)
	rospy.spin()

if __name__=='__main__':
	main()
