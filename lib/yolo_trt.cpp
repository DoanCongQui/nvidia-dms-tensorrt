#include "yolo_trt.h"

#include "config.h"
#include "nms.h"
#include "trt_logger.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

namespace face_ai {

namespace {

void checkCuda(cudaError_t status, const char* action) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(action) + ": " + cudaGetErrorString(status));
    }
}

} // namespace

YoloTRT::YoloTRT(const std::string& enginePath) {
    std::ifstream f(enginePath, std::ios::binary);
    if (!f.good()) {
        throw std::runtime_error("Engine file not found: " + enginePath);
    }

    f.seekg(0, std::ios::end);
    const size_t sz = static_cast<size_t>(f.tellg());
    f.seekg(0, std::ios::beg);

    std::vector<char> buf(sz);
    f.read(buf.data(), static_cast<std::streamsize>(sz));

    runtime_ = nvinfer1::createInferRuntime(gLogger);
    engine_ = runtime_->deserializeCudaEngine(buf.data(), sz);
    if (!engine_) {
        throw std::runtime_error("deserializeCudaEngine failed");
    }

    context_ = engine_->createExecutionContext();
    if (!context_) {
        throw std::runtime_error("createExecutionContext failed");
    }

    const int nb = engine_->getNbBindings();
    for (int i = 0; i < nb; ++i) {
        if (engine_->bindingIsInput(i)) {
            inputIndex_ = i;
        } else {
            outputIndex_ = i;
        }
    }
    if (inputIndex_ < 0 || outputIndex_ < 0) {
        throw std::runtime_error("Cannot find input/output bindings");
    }

    auto inDims0 = engine_->getBindingDimensions(inputIndex_);
    bool dynamic = false;
    for (int i = 0; i < inDims0.nbDims; ++i) {
        if (inDims0.d[i] < 0) {
            dynamic = true;
        }
    }
    if (dynamic) {
        context_->setBindingDimensions(
            inputIndex_,
            nvinfer1::Dims4{1, 3, config::YOLO_IN, config::YOLO_IN}
        );
    }

    inDims_ = context_->getBindingDimensions(inputIndex_);
    outDims_ = context_->getBindingDimensions(outputIndex_);
    inCount_ = static_cast<size_t>(volumeDims(inDims_));
    outCount_ = static_cast<size_t>(volumeDims(outDims_));

    std::cout << "[YOLO] input dims: ";
    for (int i = 0; i < inDims_.nbDims; ++i) {
        std::cout << inDims_.d[i] << (i + 1 == inDims_.nbDims ? "\n" : "x");
    }
    std::cout << "[YOLO] output dims: ";
    for (int i = 0; i < outDims_.nbDims; ++i) {
        std::cout << outDims_.d[i] << (i + 1 == outDims_.nbDims ? "\n" : "x");
    }

    checkCuda(cudaStreamCreate(&stream_), "cudaStreamCreate failed");

    for (auto& slot : slots_) {
        checkCuda(cudaMalloc(&slot.dInput, inCount_ * sizeof(float)), "cudaMalloc input failed");
        checkCuda(cudaMalloc(&slot.dOutput, outCount_ * sizeof(float)), "cudaMalloc output failed");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&slot.hInputPinned), inCount_ * sizeof(float)), "cudaMallocHost input failed");
        checkCuda(cudaMallocHost(reinterpret_cast<void**>(&slot.hOutputPinned), outCount_ * sizeof(float)), "cudaMallocHost output failed");
        checkCuda(cudaEventCreateWithFlags(&slot.done, cudaEventDisableTiming), "cudaEventCreateWithFlags failed");
    }

    bindings_.resize(engine_->getNbBindings(), nullptr);
}

YoloTRT::~YoloTRT() {
    for (auto& slot : slots_) {
        if (slot.done) {
            cudaEventDestroy(slot.done);
        }
        if (slot.dInput) {
            cudaFree(slot.dInput);
        }
        if (slot.dOutput) {
            cudaFree(slot.dOutput);
        }
        if (slot.hInputPinned) {
            cudaFreeHost(slot.hInputPinned);
        }
        if (slot.hOutputPinned) {
            cudaFreeHost(slot.hOutputPinned);
        }
    }

    if (stream_) {
        cudaStreamDestroy(stream_);
    }
    if (context_) {
        context_->destroy();
    }
    if (engine_) {
        engine_->destroy();
    }
    if (runtime_) {
        runtime_->destroy();
    }
}

bool YoloTRT::submitAsync(int slotIndex, const cv::Mat& bgr) {
    if (slotIndex < 0 || slotIndex > 1 || bgr.empty()) {
        return false;
    }

    Slot& slot = slots_[slotIndex];
    if (slot.inFlight) {
        if (cudaEventQuery(slot.done) != cudaSuccess) {
            return false;
        }
        slot.inFlight = false;
    }

    cv::Mat inp = letterbox(bgr, config::YOLO_IN, config::YOLO_IN, slot.lb);

    cv::Mat rgb;
    cv::cvtColor(inp, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);

    std::vector<cv::Mat> chw(3);
    for (int c = 0; c < 3; ++c) {
        chw[c] = cv::Mat(
            config::YOLO_IN,
            config::YOLO_IN,
            CV_32F,
            slot.hInputPinned + c * config::YOLO_IN * config::YOLO_IN
        );
    }
    cv::split(rgb, chw);

    bindings_[inputIndex_] = slot.dInput;
    bindings_[outputIndex_] = slot.dOutput;

    checkCuda(cudaMemcpyAsync(
        slot.dInput,
        slot.hInputPinned,
        inCount_ * sizeof(float),
        cudaMemcpyHostToDevice,
        stream_
    ), "cudaMemcpyAsync input failed");

    if (!context_->enqueueV2(bindings_.data(), stream_, nullptr)) {
        throw std::runtime_error("TensorRT enqueueV2 failed");
    }

    checkCuda(cudaMemcpyAsync(
        slot.hOutputPinned,
        slot.dOutput,
        outCount_ * sizeof(float),
        cudaMemcpyDeviceToHost,
        stream_
    ), "cudaMemcpyAsync output failed");

    checkCuda(cudaEventRecord(slot.done, stream_), "cudaEventRecord failed");
    slot.inFlight = true;
    return true;
}

bool YoloTRT::tryGetResult(
    int slotIndex,
    const cv::Mat& originalFrame,
    std::vector<std::pair<cv::Rect, float>>& outPhones
) {
    outPhones.clear();
    if (slotIndex < 0 || slotIndex > 1) {
        return false;
    }

    Slot& slot = slots_[slotIndex];
    if (!slot.inFlight || cudaEventQuery(slot.done) != cudaSuccess) {
        return false;
    }

    slot.inFlight = false;
    parsePhones(slot, originalFrame, outPhones);
    return true;
}

void YoloTRT::parsePhones(const Slot& slot, const cv::Mat& frame, std::vector<std::pair<cv::Rect, float>>& out) {
    const int nbDims = outDims_.nbDims;
    int a = 1;
    int b = 1;
    int c = 1;

    if (nbDims == 3) {
        a = outDims_.d[0];
        b = outDims_.d[1];
        c = outDims_.d[2];
    } else if (nbDims == 2) {
        a = outDims_.d[0];
        b = outDims_.d[1];
    } else {
        return;
    }

    int n = 0;
    bool layout84xN = false;
    if (nbDims == 3) {
        if (b == 84) {
            n = c;
            layout84xN = true;
        } else if (c == 84) {
            n = b;
        } else {
            return;
        }
    } else if (a == 84) {
        n = b;
        layout84xN = true;
    } else if (b == 84) {
        n = a;
    } else {
        return;
    }

    std::vector<cv::Rect> boxes;
    std::vector<float> scores;

    auto toOriginal = [&](float x, float y, float w, float h, float score) {
        float x1 = x - w / 2.0f;
        float y1 = y - h / 2.0f;
        float x2 = x + w / 2.0f;
        float y2 = y + h / 2.0f;

        x1 = (x1 - slot.lb.padw) / slot.lb.r;
        y1 = (y1 - slot.lb.padh) / slot.lb.r;
        x2 = (x2 - slot.lb.padw) / slot.lb.r;
        y2 = (y2 - slot.lb.padh) / slot.lb.r;

        x1 = std::max(0.0f, std::min(x1, static_cast<float>(frame.cols - 1)));
        y1 = std::max(0.0f, std::min(y1, static_cast<float>(frame.rows - 1)));
        x2 = std::max(0.0f, std::min(x2, static_cast<float>(frame.cols - 1)));
        y2 = std::max(0.0f, std::min(y2, static_cast<float>(frame.rows - 1)));

        const int ix1 = static_cast<int>(std::round(x1));
        const int iy1 = static_cast<int>(std::round(y1));
        const int ix2 = static_cast<int>(std::round(x2));
        const int iy2 = static_cast<int>(std::round(y2));

        if (ix2 > ix1 && iy2 > iy1) {
            boxes.emplace_back(ix1, iy1, ix2 - ix1, iy2 - iy1);
            scores.emplace_back(score);
        }
    };

    const float* outPtr = slot.hOutputPinned;
    for (int i = 0; i < n; ++i) {
        float x = 0.0f;
        float y = 0.0f;
        float w = 0.0f;
        float h = 0.0f;
        float best = -1.0f;
        int bestId = -1;

        if (layout84xN) {
            x = outPtr[0 * n + i];
            y = outPtr[1 * n + i];
            w = outPtr[2 * n + i];
            h = outPtr[3 * n + i];
            for (int k = 0; k < config::NUM_CLASSES; ++k) {
                const float sc = outPtr[(4 + k) * n + i];
                if (sc > best) {
                    best = sc;
                    bestId = k;
                }
            }
        } else {
            const float* det = &outPtr[i * (4 + config::NUM_CLASSES)];
            x = det[0];
            y = det[1];
            w = det[2];
            h = det[3];
            for (int k = 0; k < config::NUM_CLASSES; ++k) {
                const float sc = det[4 + k];
                if (sc > best) {
                    best = sc;
                    bestId = k;
                }
            }
        }

        if (bestId == config::PHONE_CLASS_ID && best >= config::PHONE_CONF_TH) {
            toOriginal(x, y, w, h, best);
        }
    }

    if (boxes.empty()) {
        return;
    }

    const auto keep = nmsIndices(boxes, scores, config::PHONE_NMS_TH);
    out.reserve(keep.size());
    for (int idx : keep) {
        out.push_back({boxes[idx], scores[idx]});
    }
}

} // namespace face_ai
