#include "camera_capture.h"

#include "config.h"

#include <chrono>
#include <iostream>
#include <mutex>
#include <string>
#include <thread>

namespace face_ai {

void captureThread(SharedState& state) {
    const std::string gst =
        "v4l2src device=/dev/video0 ! "
        "image/jpeg, width=640, height=480, framerate=30/1 ! "
        "nvv4l2decoder mjpeg=1 ! "
        "nvvidconv ! "
        "video/x-raw, format=BGRx ! "
        "videoconvert ! "
        "appsink drop=true sync=false";

    cv::VideoCapture cap(gst, cv::CAP_GSTREAMER);

    cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    cap.set(cv::CAP_PROP_FPS, 30);

    if (!cap.isOpened()) {
        std::cerr << "Cannot open camera in capture thread\n";
        state.running.store(false);
        return;
    }

    while (state.running.load()) {
        cv::Mat frame;
        cap >> frame;
        if (frame.empty()) {
            continue;
        }

        {
            std::lock_guard<std::mutex> lk(state.frameMutex);
            state.latestFrame = frame;
            ++state.frameSeq;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace face_ai
