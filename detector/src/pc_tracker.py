#! /usr/bin/env python

import rospy 
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2

from visualization_msgs.msg import Marker

from time import sleep


def pclcb(msg):
	width = msg.width
	height = msg.height
	for point in pc2.read_points(msg, skip_nans=True):
		pt_x = point[0]
		pt_y = point[1]
		pt_z = point[2]
		print('PCL INFO: \n	X: {0}	Y: {1}	Z: {2} \n'.format(round(pt_x,2),round(pt_y,2),round(pt_z,2)))
		sleep(2)
		print("			Width: {0} 	Height: {1}\n".format(width, height))
	
def boxcb(msg):
	x_position = msg.pose.position.x
	y_position = msg.pose.position.y
	z_position = msg.pose.position.z
	print("BOX INFO: \n	X pos: {0}	Y pos: {1}	Z pos: {2} \n".format(round(x_position,2), round(y_position,2), round(z_position, 2)))
	sleep(2)

def main():
	rospy.init_node('pointCloud_sub')
	rospy.loginfo("PCL Tracker Initialized")
	camera_sub = rospy.Subscriber('/camera/depth/points', PointCloud2, pclcb)
	marker_sub = rospy.Subscriber('/turtlebot_follower2/bbox',Marker, boxcb)
	rospy.spin()
	
if __name__=='__main__':
	main()
