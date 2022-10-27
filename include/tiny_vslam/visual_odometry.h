#include <ctype.h>

#include <algorithm>  // for copy
#include <cstddef>
#include <iostream>
#include <iterator>  // for ostream_iterator
#include <sstream>
#include <string>
#include <vector>

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/video/tracking.hpp"

#include <cv_bridge/cv_bridge.h>
#include <nav_msgs/Path.h>

#define MAX_NUM_FEAT 5000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2
#define DIRECT_MODE 0
#define FEATURE_MODE 1

std::mutex mutex_;
nav_msgs::Path path;

int Mode = 1;
bool is_init = 0;
cv::Mat prevImage, currImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;
cv::Mat prevDescriptors, currDescriptors;
cv::Mat E, R, t, R_f, t_f, mask;
double scale = 1.00;
std::vector<cv::KeyPoint> keypoints;
/* Realsense camera metrix */
// Mat cameraMatrix = (Mat1d(3, 3) << 630.1563720703125, 0.0, 642.2313232421875,
//                                              0.0, 629.5250854492188,
//                                              359.1725769042969, 0.0,
//                                              0.0, 1.0);
/* KITTI Dataset 00 camera metrix */
cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 718.856, 0.0, 607.1928, 0.0, 718.856,
                        185.2157, 0.0, 0.0, 1.0);

void featureTracking(cv::Mat img_1, cv::Mat img_2,
                     std::vector<cv::Point2f> &points1,
                     std::vector<cv::Point2f> &points2,
                     std::vector<uchar> &status) {
    std::vector<float> err;
    cv::Size winSize = cv::Size(40, 40);
    cv::Size SPwinSize = cv::Size(3, 3);  // search window size=(2*n+1,2*n+1)
    cv::Size zeroZone =
        cv::Size(1, 1);  // dead_zone size in centre=(2*n+1,2*n+1)
    cv::TermCriteria SPcriteria = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::cornerSubPix(img_1, points1, SPwinSize, zeroZone, SPcriteria);
    cv::TermCriteria termcrit = cv::TermCriteria(
        cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01);
    cv::calcOpticalFlowPyrLK(img_1, img_2, points1, points2, status, err,
                             winSize, 3, termcrit, 0, 0.0001);

    int indexCorrection = 0;
    for (int i = 0; i < status.size(); i++) {
        cv::Point2f pt = points2.at(i - indexCorrection);
        if ((status.at(i) == 0) || (pt.x < 0) || (pt.y < 0)) {
            if ((pt.x < 0) || (pt.y < 0)) {
                status.at(i) = 0;
            }
            points1.erase(points1.begin() + (i - indexCorrection));
            points2.erase(points2.begin() + (i - indexCorrection));
            indexCorrection++;
        }
    }
}

void featureDetection(cv::Mat img, std::vector<cv::Point2f> &points,
                      int feature, cv::Mat &descriptors) {
    // Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    if (feature == FEATURE_FAST) {
        // FAST algorithm
        int fast_threshold = 20;
        bool nonmaxSuppression = true;
        cv::FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    } else if (feature == FEATURE_ORB) {
        // ORB algorithm
        cv::Ptr<cv::Feature2D> orb = cv::ORB::create(MAX_NUM_FEAT);
        orb->detectAndCompute(img, cv::Mat(), keypoints, descriptors);
        cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    } else if (feature == FEATURE_SHI_TOMASI) {
        // Shi-Tomasi algorithm
        cv::goodFeaturesToTrack(img, points, MAX_NUM_FEAT, 0.01, 10);
    }
}

void findFeatureMatch(const cv::Mat img_1, const cv::Mat img_2,
                      std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_NUM_FEAT);
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    orb->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);
    std::vector<cv::DMatch> matches;
    matcher->match(descriptors_1, descriptors_2, matches);
    double minDist = 10000, maxDist = 0;
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }
    printf("-- Max dist : %f \n", maxDist);
    printf("-- Min dist : %f \n", minDist);
    // std::vector<cv::DMatch> goodmatches;
    // std::vector<cv::Point2f> pt_1, pt_2;
    points1.clear();
    points2.clear();
    // goodmatches.empty();
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * minDist, 30.0)) {
            // goodmatches.push_back(matches[i]);
            points1.push_back(keypoints_1[matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }
}

void findFeatureMatch(const cv::Mat &img_1, const cv::Mat &img_2,
                          std::vector<cv::KeyPoint> &keypoints_1,
                          std::vector<cv::KeyPoint> &keypoints_2,
                          std::vector<cv::DMatch> &matches) {
    cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_NUM_FEAT);
    cv::Mat descriptors_1, descriptors_2;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    orb->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);
    std::vector<cv::DMatch> match;
    matcher->match(descriptors_1, descriptors_2, match);

    double min_dist = 10000, max_dist = 0;

    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= std::max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

void draw_detected_image(cv::Mat &detectImage)
{
    for (unsigned int i = 0; i < currFeatures.size(); i++)
                cv::circle(detectImage, currFeatures[i], 3,  cv::Scalar(0, 255, 0), 1, 8, 0);
}

void poseEstimation() 
{
    if (Mode == DIRECT_MODE) {
        std::vector<uchar> status;
        featureDetection(prevImage, prevFeatures, FEATURE_FAST,
                            prevDescriptors);
        featureTracking(prevImage, currImage, prevFeatures,
                        currFeatures, status);
    } else
        findFeatureMatch(prevImage, currImage, prevFeatures,
                            currFeatures);
    E = cv::findEssentialMat(currFeatures, prevFeatures, cameraMatrix,
                                cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t,
                    mask);

    if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) &&
        (t.at<double>(2) > t.at<double>(1))) {
        t_f = t_f + scale * (R_f * t);
        R_f = R * R_f;
    }

    prevImage = currImage.clone();
    prevFeatures = currFeatures;
    prevDescriptors = currDescriptors.clone();
}

void initial_pose()
{
    prevImage = currImage.clone();
    R_f = cv::Mat::eye(3, 3, CV_64FC1);
    t_f = cv::Mat::zeros(3,1, CV_64FC1);
    is_init = true;
}

void publishPath(ros::Publisher& publisher)
{
    geometry_msgs::PoseStamped pose_stamped;
    pose_stamped.pose.position.x = t_f.at<double>(2);
    pose_stamped.pose.position.y = t_f.at<double>(0);
    // pose_stamped.pose.position.z = t_f.at<double>(2);

    // We don't care about the orientation
    pose_stamped.pose.orientation.w = 1;
    pose_stamped.header.stamp = ros::Time::now();
    pose_stamped.header.frame_id="world";
    path.poses.push_back(pose_stamped);
    publisher.publish(path);
}