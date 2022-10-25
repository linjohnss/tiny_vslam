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

#define MAX_NUM_FEAT 5000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2
#define DIRECT_MODE 0
#define FEATURE_MODE 1

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

cv::Point2d pixel2cam(const cv::Point2d &p, const cv::Mat &K) {
    return cv::Point2d
    (
        (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
        (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
    );
}

void create3DPoint(cv::Mat img_depth, std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point3f> &point3D, cv::Mat cameraMatrix) {
    point3D.clear();
    for (cv::Point2d p:points1) {
        ushort d = img_depth.ptr<unsigned short>(int(p.y))[int(p.x)];
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;
        cv::Point2d p1 = pixel2cam(p, cameraMatrix);
        point3D.push_back(cv::Point3f(p1.x * dd, p1.y * dd, dd));
    }
}