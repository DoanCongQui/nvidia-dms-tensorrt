# 🚗 NVIDIA DMS TensorRT

**NVIDIA DMS TensorRT** là hệ thống Driver Monitoring System (DMS) chạy thời gian thực, dùng **OpenCV**, **YuNet**, **YOLO**, **CUDA** và **TensorRT** để theo dõi trạng thái người lái.

Dự án được tối ưu theo hướng chạy trên thiết bị NVIDIA/Jetson hoặc máy Linux có GPU NVIDIA, tập trung vào tốc độ xử lý camera realtime và suy luận AI bằng TensorRT engine.

## ✨ Tính Năng

- 🧑 Phát hiện khuôn mặt bằng YuNet.
- 🧭 Ước lượng hướng quay đầu của người lái.
- 👁️ Ước lượng hướng nhìn dựa trên vùng mắt.
- 😴 Phát hiện nhắm mắt / dấu hiệu buồn ngủ.
- 📱 Phát hiện điện thoại bằng YOLO TensorRT.
- ⚡ Chạy YOLO bằng CUDA/TensorRT để tăng tốc inference.
- 🧵 Tách luồng camera và luồng YOLO riêng để giảm độ trễ.
- 🎥 Hiển thị realtime bằng OpenCV.

## 📁 Cấu Trúc Project

```text
nvidia-dms-tensorrt/
|-- CMakeLists.txt
|-- README.md
|-- models/
|   |-- face_detection_yunet_2023mar.onnx
|   |-- face_detection_yunet_2023mar_int8.onnx
|   |-- yolo_model.onnx
|   `-- yolo_model.engine
|-- src/
|   `-- main.cpp
`-- lib/
    |-- camera_capture.h/.cpp
    |-- config.h
    |-- dms.h/.cpp
    |-- letterbox.h/.cpp
    |-- nms.h/.cpp
    |-- shared_state.h/.cpp
    |-- trt_logger.h/.cpp
    |-- yolo_trt.h/.cpp
    `-- yolo_worker.h/.cpp
```

## 🧩 Ý Nghĩa Các Module

| Module | Chức năng |
|---|---|
| `src/main.cpp` | Entry point, vòng lặp chính, hiển thị UI realtime |
| `lib/config.h` | Cấu hình đường dẫn model, ngưỡng cảnh báo, input size |
| `lib/camera_capture.*` | Đọc camera bằng OpenCV/GStreamer |
| `lib/yolo_trt.*` | Load TensorRT engine, cấp phát CUDA buffer, chạy YOLO async |
| `lib/yolo_worker.*` | Luồng riêng để chạy YOLO, tránh nghẽn camera |
| `lib/dms.*` | Xử lý head pose, gaze, eye closure |
| `lib/letterbox.*` | Resize ảnh kiểu letterbox cho YOLO |
| `lib/nms.*` | Non-Maximum Suppression cho bounding box |
| `lib/shared_state.*` | Trạng thái dùng chung giữa các thread |
| `lib/trt_logger.*` | Logger cho TensorRT |

## 🧰 Yêu Cầu Hệ Thống

Cần cài đặt trước:

- ✅ C++17 compiler
- ✅ CMake 3.16+
- ✅ OpenCV có `FaceDetectorYN` / OpenCV contrib
- ✅ CUDA
- ✅ TensorRT
- ✅ GStreamer nếu dùng camera pipeline trên Jetson/Linux

Khuyến nghị chạy trên:

- NVIDIA Jetson Nano / Xavier / Orin
- Linux có GPU NVIDIA
- Môi trường đã cài CUDA và TensorRT đúng phiên bản

## 📦 Model Cần Có

Thư mục `models/` cần có:

```text
models/
|-- face_detection_yunet_2023mar.onnx
|-- yolo_model.onnx
`-- yolo_model.engine
```

Trong đó:

- 🧑 `face_detection_yunet_2023mar.onnx`: model YuNet để phát hiện khuôn mặt.
- 🧠 `yolo_model.onnx`: model YOLO dạng ONNX.
- ⚡ `yolo_model.engine`: TensorRT engine được build từ file ONNX.

> Lưu ý: file `.engine` phụ thuộc vào GPU, CUDA, TensorRT và kiến trúc máy. Không nên copy `.engine` từ máy khác. Nên tạo lại engine trực tiếp trên máy sẽ chạy ứng dụng.

### Trạng thái model trong workspace hiện tại

- `models/yolo_model.onnx` đã có.
- `models/yolo_model.engine` chưa có, cần tạo bằng `trtexec`.
- `models/face_detection_yunet_2023mar.onnx` đang là Git LFS pointer. Sau khi clone repo, cần chạy `git lfs pull` hoặc tải model YuNet thật trước khi chạy ứng dụng.

## 🧭 Đường Dẫn Model

Đường dẫn model hiện được cấu hình trong `lib/config.h`:

```cpp
inline constexpr const char* TRT_ENGINE_PATH = "../models/yolo_model.engine";
inline constexpr const char* YUNET_MODEL_PATH = "../models/face_detection_yunet_2023mar.onnx";
```

Ứng dụng nên được chạy từ thư mục `build/` để đường dẫn `../models/...` hoạt động đúng:

```bash
cd nvidia-dms-tensorrt
cmake -S . -B build
cmake --build build
cd build
./face_ai
```

Khi chạy từ `build/`, ứng dụng sẽ tìm model tại:

```text
nvidia-dms-tensorrt/models/yolo_model.engine
nvidia-dms-tensorrt/models/face_detection_yunet_2023mar.onnx
```

## ⚡ Tạo TensorRT Engine Từ ONNX

Nếu đã có:

```text
models/yolo_model.onnx
```

tạo file engine bằng `trtexec`:

```bash
cd nvidia-dms-tensorrt
trtexec \
  --onnx=models/yolo_model.onnx \
  --saveEngine=models/yolo_model.engine \
  --fp16
```

Nếu TensorRT yêu cầu khai báo workspace, dùng một trong hai cách sau tùy phiên bản TensorRT:

```bash
trtexec \
  --onnx=models/yolo_model.onnx \
  --saveEngine=models/yolo_model.engine \
  --fp16 \
  --workspace=2048
```

hoặc:

```bash
trtexec \
  --onnx=models/yolo_model.onnx \
  --saveEngine=models/yolo_model.engine \
  --fp16 \
  --memPoolSize=workspace:2048
```

Kiểm tra file engine sau khi tạo:

```bash
ls -lh models/yolo_model.engine
```

## 🧠 Export YOLO Sang ONNX

Nếu model YOLO ban đầu là file `.pt`, export sang ONNX trước:

```bash
yolo export model=yolo_model.pt format=onnx imgsz=640 opset=12 simplify=True
```

Sau đó copy hoặc đổi tên file ONNX thành:

```text
models/yolo_model.onnx
```

rồi tạo TensorRT engine:

```bash
trtexec --onnx=models/yolo_model.onnx --saveEngine=models/yolo_model.engine --fp16
```

## 🛠️ Build Project

Từ thư mục gốc project:

```bash
cmake -S . -B build
cmake --build build
```

Nếu build thành công, file chạy nằm tại:

```text
build/face_ai
```

## ▶️ Chạy Ứng Dụng

Chạy từ thư mục `build/`:

```bash
cd build
./face_ai
```

Nhấn `ESC` để thoát ứng dụng.

## ⚙️ Cấu Hình Quan Trọng

Các cấu hình chính nằm trong `lib/config.h`.

| Tham số | Ý nghĩa |
|---|---|
| `TRT_ENGINE_PATH` | Đường dẫn file TensorRT engine |
| `YUNET_MODEL_PATH` | Đường dẫn model YuNet |
| `YOLO_IN` | Kích thước input YOLO, mặc định `640` |
| `PHONE_CLASS_ID` | Class ID điện thoại trong COCO, mặc định `67` |
| `PHONE_CONF_TH` | Ngưỡng confidence khi phát hiện điện thoại |
| `PHONE_NMS_TH` | Ngưỡng NMS cho box điện thoại |
| `YOLO_EVERY_N` | Số frame mới chạy YOLO một lần |
| `DET_W`, `DET_H` | Kích thước input YuNet |
| `EYE_CLOSE_TH` | Thời gian nhắm mắt bắt đầu cảnh báo |
| `EYE_CLOSE_STRONG` | Thời gian nhắm mắt cảnh báo mạnh |
| `TH_LOOK_AWAY` | Thời gian nhìn lệch trước khi cảnh báo |
| `TH_PHONE` | Thời gian phát hiện điện thoại trước khi cảnh báo |
| `TH_NO_FACE` | Thời gian không thấy mặt trước khi cảnh báo |

Nếu FPS thấp, có thể tăng:

```cpp
inline constexpr int YOLO_EVERY_N = 2;
```

hoặc:

```cpp
inline constexpr int YOLO_EVERY_N = 3;
```

Giá trị càng cao thì YOLO chạy ít hơn, FPS có thể tốt hơn nhưng phát hiện điện thoại sẽ chậm hơn.

## 🎥 Camera

Camera hiện được cấu hình trong `lib/camera_capture.cpp` bằng GStreamer pipeline:

```cpp
v4l2src device=/dev/video0 !
image/jpeg, width=640, height=480, framerate=30/1 !
nvv4l2decoder mjpeg=1 !
nvvidconv !
video/x-raw, format=BGRx !
videoconvert !
appsink drop=true sync=false
```

Nếu không dùng Jetson/GStreamer, có thể đổi về camera mặc định của OpenCV:

```cpp
cv::VideoCapture cap(0);
```

## 🧯 Lỗi Thường Gặp

### ❌ Không tìm thấy `yolo_model.engine`

Kiểm tra khi đang đứng trong thư mục `build/`:

```bash
ls -lh ../models/yolo_model.engine
```

Nếu chưa có file engine, tạo lại bằng `trtexec`.

### 🧑 Không tạo được `FaceDetectorYN`

Nguyên nhân thường gặp:

- OpenCV chưa có module `FaceDetectorYN`.
- OpenCV thiếu contrib module.
- Phiên bản OpenCV quá cũ.

Cách xử lý: cài OpenCV contrib hoặc build OpenCV bản mới có module `objdetect`.

### ⚡ TensorRT engine bị lỗi khi load

Thường do file `.engine` được tạo trên máy khác hoặc khác phiên bản CUDA/TensorRT/GPU.

Cách xử lý:

1. Xóa file `models/yolo_model.engine` cũ.
2. Tạo lại engine trên đúng máy sẽ chạy ứng dụng.
3. Đảm bảo ONNX input phù hợp với `YOLO_IN = 640`.

### 🐢 FPS thấp

Có thể thử:

- Tăng `YOLO_EVERY_N`.
- Dùng engine `--fp16`.
- Giảm độ phân giải camera.
- Đảm bảo đang chạy bằng TensorRT engine, không phải ONNX trực tiếp.

## 📌 Trạng Thái Project

Project đã được tách module để dễ bảo trì:

- 🚀 `src/`: chứa entry point.
- 🧩 `lib/`: chứa các module xử lý riêng.
- 📦 `models/`: chứa model ONNX và TensorRT engine.
- 🛠️ `CMakeLists.txt`: quản lý build.

## 📄 License

Chưa khai báo license.
