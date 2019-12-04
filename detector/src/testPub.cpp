#include <ros/ros.h>
#include "std_msgs/String.h"
#include <sstream>

int main(int argc, char **argv){
	//The init function? sorry I'm bad at documenting code. Will fix later.
	ros::init(argc, argv, "Talker");

	// NodeHandle is the main access point to communication with ROS. This handler
	// initializes the main node? perhaps, who knows? not me thats for sure.
	ros::NodeHandle n;

	//The advertise function tells ROS that you want to pusblish a topic
	ros::Publisher chatter_pub = n.advertise<std_msgs::String>("Ground",1000);
	ros::Rate loop_rate(10); // A publishing rate of 10Hz? Good job ROS wiki.

	int count = 0; //To count the number of messages being sent, cause why not?
	//I like the function name to check that ros has not been shutdown, kudos to whoever named it.
	while(ros::ok()){
		std_msgs::String msg;
		std::stringstream ss;
		ss<< "Bee bop " << count;
		msg.data = ss.str();
		ROS_INFO("%s", msg.data.c_str());
		chatter_pub.publish(msg);

		ros::spinOnce();
		loop_rate.sleep();
		++count;
	}
	return 0;
}
