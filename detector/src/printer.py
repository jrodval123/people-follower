#! /usr/bin/env python

import rospy

def main():
	rospy.init_node('printer_node')
	print('\n Hello? \n\n')
	rospy.spin()

if __name__=='__main__':
	main()
