#include "trt_logger.h"

#include <iostream>

namespace face_ai {

Logger gLogger;

void Logger::log(Severity severity, const char* msg) noexcept {
    if (severity <= Severity::kWARNING) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

int64_t volumeDims(const nvinfer1::Dims& d) {
    int64_t v = 1;
    for (int i = 0; i < d.nbDims; ++i) {
        v *= d.d[i];
    }
    return v;
}

} // namespace face_ai
