#pragma once

#include <opencv2/opencv.hpp>

#include <vector>

namespace face_ai {

float iou(const cv::Rect& a, const cv::Rect& b);
std::vector<int> nmsIndices(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nmsTh);

} // namespace face_ai
