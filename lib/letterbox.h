#pragma once

#include <opencv2/opencv.hpp>

namespace face_ai {

struct LetterBoxInfo {
    float r = 1.0f;
    int padw = 0;
    int padh = 0;
};

cv::Mat letterbox(const cv::Mat& src, int newW, int newH, LetterBoxInfo& info);

} // namespace face_ai
