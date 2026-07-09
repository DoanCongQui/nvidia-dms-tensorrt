#pragma once

namespace face_ai::config {

inline constexpr const char* TRT_ENGINE_PATH = "../models/yolo_model.engine";
inline constexpr const char* YUNET_MODEL_PATH = "../models/face_detection_yunet_2023mar.onnx";
inline constexpr int CAM_ID = 0;

inline constexpr int YOLO_IN = 640;
inline constexpr int NUM_CLASSES = 80;
inline constexpr int PHONE_CLASS_ID = 67;
inline constexpr float PHONE_CONF_TH = 0.40f;
inline constexpr float PHONE_NMS_TH = 0.45f;
inline constexpr int YOLO_EVERY_N = 1;

inline constexpr int DET_W = 320;
inline constexpr int DET_H = 320;

inline constexpr float HEAD_LEFT_DEG = 18.0f;
inline constexpr float HEAD_RIGHT_DEG = -18.0f;
inline constexpr float HEAD_STRONG_LEFT_DEG = 28.0f;
inline constexpr float HEAD_STRONG_RIGHT_DEG = -28.0f;

inline constexpr float GAZE_LEFT_TH = 0.42f;
inline constexpr float GAZE_RIGHT_TH = 0.58f;

inline constexpr float EYE_ROI_W = 0.28f;
inline constexpr float EYE_ROI_H = 0.18f;

inline constexpr float ALPHA_YAW = 0.25f;
inline constexpr float ALPHA_GAZE = 0.35f;

inline constexpr float EYE_CLOSE_TH = 1.0f;
inline constexpr float EYE_CLOSE_STRONG = 2.0f;

inline constexpr float TH_LOOK_AWAY = 2.0f;
inline constexpr float TH_PHONE = 1.0f;
inline constexpr float TH_NO_FACE = 3.0f;

} // namespace face_ai::config
