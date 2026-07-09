#pragma once

#include "shared_state.h"
#include "yolo_trt.h"

namespace face_ai {

void yoloThread(SharedState& state, YoloTRT& yolo);

} // namespace face_ai
