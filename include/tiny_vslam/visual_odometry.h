#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/video/tracking.hpp"
#include "opencv2/optflow/motempl.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <cv_bridge/cv_bridge.h>
#include <Eigen/Core>
#include <Eigen/Geometry>
// #include <time.h>

#define MAX_NUM_FEAT 5000
#define MIN_NUM_FEAT 2000

#define FEATURE_FAST 0
#define FEATURE_ORB 1
#define FEATURE_SHI_TOMASI 2
#define DIRECT_MODE 0
#define FEATURE_MODE 1

int Mode = 1;
bool is_init = 0;
cv::Mat prevImage, currImage;
std::vector<cv::Point2f> prevFeatures, currFeatures;
cv::Mat prevDescriptors, currDescriptors;
cv::Mat E, R, t, mask;
Eigen::Matrix3d R_f;
Eigen::Vector3d t_f;
double scale = 0.20;
std::vector<cv::KeyPoint> keypoints;
/* Realsense camera metrix */
// cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 632.9028930664062, 0.0, 642.2313232421875,
//                                             0.0, 632.2689208984375, 359.1725769042969,
//                                             0.0, 0.0, 1.0);

/* KITTI Dataset 00 camera metrix */
cv::Mat cameraMatrix = (cv::Mat1d(3, 3) << 718.856, 0.0, 607.1928,
                                            0.0, 718.856, 185.2157,
                                            0.0, 0.0, 1.0);

static void cal_diff()
{
    cv::Mat silf;
    cv::absdiff(prevImage, currImage, silf);
    cv::threshold(silf, silf, 100, 255, cv::THRESH_BINARY);
    cv::imshow("Img", silf);
    cv::waitKey(1);
}


void featureTracking(cv::Mat img_1, cv::Mat img_2, std::vector<cv::Point2f> &points1, 
                        std::vector<cv::Point2f> &points2, std::vector<uchar> &status) 
{
    std::vector<float> err;
    cv::Size winSize = cv::Size(40, 40);
    cv::Size SPwinSize = cv::Size(3, 3);  // search window size=(2*n+1,2*n+1)
    cv::Size zeroZone = cv::Size(1, 1);  // dead_zone size in centre=(2*n+1,2*n+1)
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

void featureDetection(cv::Mat img, std::vector<cv::Point2f> &points, int feature)
{
    // Mat descriptors;
    std::vector<cv::KeyPoint> keypoints;
    if (feature == FEATURE_FAST) {
        // FAST algorithm
        int fast_threshold = 30;
        bool nonmaxSuppression = true;
        cv::FAST(img, keypoints, fast_threshold, nonmaxSuppression);
        cv::KeyPoint::convert(keypoints, points, std::vector<int>());
    } else if (feature == FEATURE_SHI_TOMASI) {
        // Shi-Tomasi algorithm
        cv::goodFeaturesToTrack(img, points, MAX_NUM_FEAT, 0.01, 10);
    }
}

void findFeatureMatch(const cv::Mat img_1, const cv::Mat img_2,
                      std::vector<cv::Point2f> &points1,
                      std::vector<cv::Point2f> &points2)
{
    cv::Ptr<cv::ORB> orb = cv::ORB::create(MAX_NUM_FEAT);
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    orb->detectAndCompute(img_1, cv::Mat(), keypoints_1, descriptors_1);
    orb->detectAndCompute(img_2, cv::Mat(), keypoints_2, descriptors_2);
    if (descriptors_1.empty() || descriptors_2.empty()) {
        points1.clear();
        points2.clear();
        return;
    }

    std::vector<cv::DMatch> matches;
    try {
        matcher->match(descriptors_1, descriptors_2, matches);
    }
    catch(const std::exception& e) {
        std::cerr << e.what() << '\n';
    }
    double minDist = 10000, maxDist = 0;
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = matches[i].distance;
        if (dist < minDist) minDist = dist;
        if (dist > maxDist) maxDist = dist;
    }

    points1.clear();
    points2.clear();
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (matches[i].distance <= std::max(2 * minDist, 30.0)) {
            points1.push_back(keypoints_1[matches[i].queryIdx].pt);
            points2.push_back(keypoints_2[matches[i].trainIdx].pt);
        }
    }
}

void draw_detected_image(cv::Mat &detectImage)
{
    for (unsigned int i = 0; i < currFeatures.size(); i++)
        cv::circle(detectImage, currFeatures[i], 3, cv::Scalar(0, 255, 0), 1, 8, 0);
}

bool poseEstimation(Eigen::Matrix3d &R_output, Eigen::Vector3d &t_output) 
{
    bool successed = true;
    if (Mode == DIRECT_MODE) {
        std::vector<uchar> status;
        featureDetection(prevImage, prevFeatures, FEATURE_FAST);
        if (prevFeatures.size() > 100) {
            featureTracking(prevImage, currImage, prevFeatures, currFeatures, status);
        }
    } else
        findFeatureMatch(prevImage, currImage, prevFeatures, currFeatures);
    if (currFeatures.size() < 200 || prevFeatures.size() < 200) {
        is_init = false;
        return false;
    }
    cal_diff();
    E = cv::findEssentialMat(currFeatures, prevFeatures, cameraMatrix,
                                cv::RANSAC, 0.999, 1.0, mask);
    cv::recoverPose(E, currFeatures, prevFeatures, cameraMatrix, R, t, mask);
    if ((scale > 0.1) && (t.at<double>(2) > t.at<double>(0)) &&
        (t.at<double>(2) > t.at<double>(1))) {
        Eigen::Map<Eigen::Matrix3d> eigenR(R.ptr<double>(), R.rows, R.cols);
        Eigen::Map<Eigen::Vector3d> eigenT(t.ptr<double>(), t.rows, t.cols);
        t_f = t_f + scale * (R_f * eigenT);
        R_f = eigenR * R_f;
    }
    
    prevImage = currImage.clone();
    prevFeatures = currFeatures;
    prevDescriptors = currDescriptors.clone();
    R_output = R_f;
    t_output = t_f;
    return true;
}

void initial_pose()
{
    prevImage = currImage.clone();
    R_f = Eigen::Matrix3d::Identity();
    t_f = Eigen::Vector3d::Zero();
    is_init = true;
}