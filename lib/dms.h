#pragma once

#include <opencv2/opencv.hpp>

#include <array>
#include <string>

namespace face_ai {

bool solveHeadPoseYaw(
    const std::array<cv::Point2f, 5>& kps,
    const cv::Mat& cam,
    const cv::Mat& dist,
    float& yawDeg
);

float pupilXNorm(const cv::Mat& eyeBgr);
std::string gazeFromX(float x);
std::string fuseHeadEye(float yaw, bool hasYaw, const std::string& gaze);

} // namespace face_ai
