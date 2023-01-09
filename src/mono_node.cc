#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "tiny_vslam/visual_odometry.h"
#include <tf/transform_broadcaster.h>
#include <nav_msgs/Path.h>

void publishPath(nav_msgs::Path &path, Eigen::Matrix3d R_output, Eigen::Vector3d t_output)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = t_output(0);
    pose_stamped.pose.position.y = t_output(1);
    pose_stamped.pose.position.z = t_output(2);

    pose_stamped.pose.orientation.w = 1;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id="world";
    path.poses.push_back(pose_stamped);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "mono_node");
    ros::NodeHandle nh;
    std::string image_topic;
    bool is_image_compressed;
    nh.getParam("/image_topic", image_topic);
    nh.getParam("/is_image_compressed", is_image_compressed);
    image_transport::ImageTransport it(nh);
    image_transport::TransportHints hints(is_image_compressed ? "compressed" : "raw");
    cv_bridge::CvImagePtr detectImage;
    auto imageCallback = [&detectImage](const sensor_msgs::ImageConstPtr& msg) -> void {
        detectImage = cv_bridge::toCvCopy(msg, sensor_msgs::image_encodings::BGR8);
        cv::cvtColor(detectImage->image, currImage, cv::COLOR_BGR2GRAY);
        image_get = true;
    };
    image_transport::Subscriber image_subscriber = it.subscribe(image_topic, 5, imageCallback, ros::VoidPtr(), hints);
    image_transport::Publisher image_publisher = it.advertise("tiny/detected_image", 1);
    Eigen::Matrix3d R_output;
    Eigen::Vector3d t_output;
    nav_msgs::Path path;
    static tf::TransformBroadcaster br;
    bool image_get = false;
    ros::Publisher vslam_path_publisher = nh.advertise<nav_msgs::Path>("tiny/vslam_path", 1);
    path.header.stamp=ros::Time::now();
    path.header.frame_id="world";
    ros::Rate loop_rate(20);
    ROS_INFO("start main loop.");
    while (ros::ok()) {
        if (image_get) {
            if (!is_init) {
                initial_pose();
            } else if(poseEstimation(R_output, t_output)) {
                draw_detected_image(detectImage->image);
                image_publisher.publish(detectImage->toImageMsg());
                publishPath(path, R_output, t_output);
                tf::Transform transform;
                transform.setBasis(tf::Matrix3x3(R_output(0, 0), R_output(0, 1), R_output(0, 2),
                                                R_output(1, 0), R_output(1, 1), R_output(1, 2),
                                                R_output(2, 0), R_output(2, 1), R_output(2, 2)));
                transform.setOrigin(tf::Vector3(t_output(0), t_output(1), t_output(2)));
                br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cur_pos"));
                vslam_path_publisher.publish(path);
            }
            else
                ROS_WARN("Lose track!");
            image_get = false;
        }
        ros::spinOnce();
        loop_rate.sleep();
    }
}
