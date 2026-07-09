#include "dms.h"

#include "config.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace face_ai {

bool solveHeadPoseYaw(
    const std::array<cv::Point2f, 5>& kps,
    const cv::Mat& cam,
    const cv::Mat& dist,
    float& yawDeg
) {
    static const cv::Mat model3D = (cv::Mat_<double>(5, 3) <<
        -30.0, 30.0, -30.0,
         30.0, 30.0, -30.0,
          0.0,  0.0,   0.0,
        -25.0,-30.0, -30.0,
         25.0,-30.0, -30.0
    );

    cv::Mat img2D(5, 2, CV_64F);
    for (int i = 0; i < 5; ++i) {
        img2D.at<double>(i, 0) = kps[i].x;
        img2D.at<double>(i, 1) = kps[i].y;
    }

    cv::Mat rvec;
    cv::Mat tvec;
    bool ok = false;
    try {
        ok = cv::solvePnP(model3D, img2D, cam, dist, rvec, tvec, false, cv::SOLVEPNP_EPNP);
    } catch (...) {
        ok = false;
    }
    if (!ok) {
        return false;
    }

    cv::Mat r;
    cv::Rodrigues(rvec, r);

    const double sy = std::sqrt(
        r.at<double>(0, 0) * r.at<double>(0, 0) +
        r.at<double>(1, 0) * r.at<double>(1, 0)
    );
    const double yaw = std::atan2(-r.at<double>(2, 0), sy);
    yawDeg = static_cast<float>(yaw * 180.0 / CV_PI);
    return true;
}

float pupilXNorm(const cv::Mat& eyeBgr) {
    if (eyeBgr.empty()) {
        return -1.0f;
    }

    cv::Mat gray;
    cv::cvtColor(eyeBgr, gray, cv::COLOR_BGR2GRAY);
    cv::equalizeHist(gray, gray);
    cv::GaussianBlur(gray, gray, cv::Size(5, 5), 0);

    cv::Mat bw;
    cv::threshold(gray, bw, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(bw, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);
    if (contours.empty()) {
        return -1.0f;
    }

    auto it = std::max_element(contours.begin(), contours.end(), [](const auto& a, const auto& b) {
        return cv::contourArea(a) < cv::contourArea(b);
    });
    if (cv::contourArea(*it) < 15) {
        return -1.0f;
    }

    const cv::Moments m = cv::moments(*it);
    if (std::fabs(m.m00) < 1e-6) {
        return -1.0f;
    }

    const float cx = static_cast<float>(m.m10 / m.m00);
    return cx / static_cast<float>(eyeBgr.cols);
}

std::string gazeFromX(float x) {
    if (x < 0.0f) {
        return "UNKNOWN";
    }
    if (x < config::GAZE_LEFT_TH) {
        return "LEFT";
    }
    if (x > config::GAZE_RIGHT_TH) {
        return "RIGHT";
    }
    return "CENTER";
}

std::string fuseHeadEye(float yaw, bool hasYaw, const std::string& gaze) {
    if (!hasYaw) {
        if (gaze == "LEFT" || gaze == "RIGHT") {
            return gaze;
        }
        return "UNKNOWN";
    }
    if (yaw >= config::HEAD_STRONG_LEFT_DEG) {
        return "LEFT";
    }
    if (yaw <= config::HEAD_STRONG_RIGHT_DEG) {
        return "RIGHT";
    }
    if (yaw >= config::HEAD_LEFT_DEG) {
        return "LEFT";
    }
    if (yaw <= config::HEAD_RIGHT_DEG) {
        return "RIGHT";
    }
    if (gaze == "LEFT" || gaze == "RIGHT") {
        return gaze;
    }
    return "STRAIGHT";
}

} // namespace face_ai
