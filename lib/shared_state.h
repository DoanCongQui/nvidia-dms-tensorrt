#pragma once

#include <opencv2/opencv.hpp>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <utility>
#include <vector>

namespace face_ai {

struct SharedState {
    std::atomic<bool> running{true};

    cv::Mat latestFrame;
    std::mutex frameMutex;
    uint64_t frameSeq = 0;

    bool phoneDetected = false;
    std::vector<std::pair<cv::Rect, float>> phoneBoxes;
    std::mutex phoneMutex;
    std::atomic<uint64_t> yoloSeq{0};
};

} // namespace face_ai
