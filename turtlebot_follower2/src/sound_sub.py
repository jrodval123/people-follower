#! /usr/bin/env python
import rospy
from std_msgs.msg import String
from std_msgs.msg import Bool

def msgCallback(msg):
	print(msg.data)
	print("Roboooot please wake up")

def secondCallback(msg):
	print(msg.data)

def main():
	rospy.init_node('sound_suscriber')
	sub = rospy.Subscriber('/soundfxs', String, secondCallback)
	rospy.spin()

if __name__=="__main__":
	main()
