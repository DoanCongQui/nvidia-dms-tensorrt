#pragma once

#include <NvInfer.h>

#include <cstdint>

namespace face_ai {

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

extern Logger gLogger;

int64_t volumeDims(const nvinfer1::Dims& d);

} // namespace face_ai
