#include "letterbox.h"

#include <algorithm>
#include <cmath>

namespace face_ai {

cv::Mat letterbox(const cv::Mat& src, int newW, int newH, LetterBoxInfo& info) {
    const int w = src.cols;
    const int h = src.rows;
    const float r = std::min(static_cast<float>(newW) / w, static_cast<float>(newH) / h);
    const int unpadW = static_cast<int>(std::round(w * r));
    const int unpadH = static_cast<int>(std::round(h * r));
    const int dw = (newW - unpadW) / 2;
    const int dh = (newH - unpadH) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(unpadW, unpadH));

    cv::Mat out(newH, newW, src.type(), cv::Scalar(114, 114, 114));
    resized.copyTo(out(cv::Rect(dw, dh, unpadW, unpadH)));

    info.r = r;
    info.padw = dw;
    info.padh = dh;
    return out;
}

} // namespace face_ai
