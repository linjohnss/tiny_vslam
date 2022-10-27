#include <ros/ros.h>
#include <image_transport/image_transport.h>

#include <opencv2/highgui/highgui.hpp>

#include "tiny_vslam/visual_odometry.h"
#include <Eigen/Dense>

bool image_get = false;

int main(int argc, char **argv) {
    ros::init(argc, argv, "mono_node");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    bool is_image_compressed = true;
    image_transport::TransportHints hints(is_image_compressed ? "compressed" : "raw");
    cv_bridge::CvImagePtr detectImage;

    auto imageCallback = [&detectImage](const sensor_msgs::ImageConstPtr& msg) -> void {
        detectImage = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(detectImage->image, currImage, cv::COLOR_BGR2GRAY);
        image_get = true;
    };

    image_transport::Subscriber image_subscriber = it.subscribe("/image_raw0", 5, imageCallback, ros::VoidPtr(), hints);
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
                draw_detected_image(detectImage->image);
                image_publisher.publish(detectImage->toImageMsg());
                publishPath(vslam_path_publisher);
            }
            image_get = false;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}
