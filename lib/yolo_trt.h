#pragma once

#include "letterbox.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <opencv2/opencv.hpp>

#include <string>
#include <utility>
#include <vector>

namespace face_ai {

class YoloTRT {
public:
    explicit YoloTRT(const std::string& enginePath);
    ~YoloTRT();

    YoloTRT(const YoloTRT&) = delete;
    YoloTRT& operator=(const YoloTRT&) = delete;

    bool submitAsync(int slot, const cv::Mat& bgr);
    bool tryGetResult(int slot, const cv::Mat& originalFrame, std::vector<std::pair<cv::Rect, float>>& outPhones);

private:
    struct Slot {
        void* dInput = nullptr;
        void* dOutput = nullptr;
        float* hInputPinned = nullptr;
        float* hOutputPinned = nullptr;
        cudaEvent_t done{};
        bool inFlight = false;
        LetterBoxInfo lb{};
    };

    void parsePhones(const Slot& slot, const cv::Mat& frame, std::vector<std::pair<cv::Rect, float>>& out);

    nvinfer1::IRuntime* runtime_ = nullptr;
    nvinfer1::ICudaEngine* engine_ = nullptr;
    nvinfer1::IExecutionContext* context_ = nullptr;

    int inputIndex_ = -1;
    int outputIndex_ = -1;
    nvinfer1::Dims inDims_{};
    nvinfer1::Dims outDims_{};

    size_t inCount_ = 0;
    size_t outCount_ = 0;
    std::vector<void*> bindings_;
    cudaStream_t stream_{};

    Slot slots_[2];
};

} // namespace face_ai
