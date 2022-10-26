#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>

#include "tiny_vslam/visual_odometry.h"
#include <Eigen/Dense>


cv::Mat traj = cv::Mat::zeros(1000, 1000, CV_8UC3);
char text[100];
bool image_get = false;

void imageCallback(const sensor_msgs::CompressedImage::ConstPtr &msg) {
    try {
        detectImage = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(detectImage->image, currImage, cv::COLOR_BGR2GRAY);
        image_get = true;
    } catch (cv_bridge::Exception &e) {
        ROS_ERROR("Error.");
    }
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "mono_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    ros::Subscriber image_subscriber = nh.subscribe("/image_raw0/compressed", 5, imageCallback);
    image_transport::Publisher image_publisher = it.advertise("tiny/detected_image", 1);

    ros::Publisher vslam_path_publisher = nh.advertise<nav_msgs::Path>("tiny/vslam_path", 1);
    path.header.stamp=ros::Time::now();
    path.header.frame_id="world";
    ros::Rate loop_rate(20);
    ROS_INFO("start main loop.");
    while (ros::ok()) {
        if (image_get) {
            if (!is_init) {
                initial_pose();
            } else {
                poseEstimation();
                image_publisher.publish(detectImage->toImageMsg());
                publishPath(vslam_path_publisher);
            }
            image_get = false;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}
