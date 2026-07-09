#include "camera_capture.h"
#include "config.h"
#include "dms.h"
#include "shared_state.h"
#include "yolo_trt.h"
#include "yolo_worker.h"

#include <opencv2/objdetect/face.hpp>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace {

cv::Rect clampFaceRect(int x, int y, int w, int h, int frameW, int frameH) {
    const int x1 = std::max(0, std::min(frameW - 1, x));
    const int y1 = std::max(0, std::min(frameH - 1, y));
    const int x2 = std::max(0, std::min(frameW, x + w));
    const int y2 = std::max(0, std::min(frameH, y + h));
    return cv::Rect(x1, y1, std::max(0, x2 - x1), std::max(0, y2 - y1));
}

} // namespace

int main() {
    using namespace face_ai;

    cv::Ptr<cv::FaceDetectorYN> yunet;
    try {
        yunet = cv::FaceDetectorYN::create(
            config::YUNET_MODEL_PATH,
            "",
            cv::Size(config::DET_W, config::DET_H),
            0.6f,
            0.3f,
            1
        );
    } catch (...) {
        std::cerr << "Failed to create FaceDetectorYN. OpenCV-contrib with FaceDetectorYN is required.\n";
        return -1;
    }

    YoloTRT yolo(config::TRT_ENGINE_PATH);

    SharedState state;
    std::thread tCap(captureThread, std::ref(state));
    std::thread tYolo(yoloThread, std::ref(state), std::ref(yolo));

    float yawSmooth = 0.0f;
    bool yawInit = false;
    float gazeSmooth = -1.0f;
    bool gazeInit = false;

    float eyeClosedSeconds = 0.0f;
    bool eyeClosed = false;

    float tLookAway = 0.0f;
    float tPhone = 0.0f;
    float tNoFace = 0.0f;

    auto tPrev = std::chrono::steady_clock::now();
    float fps = 0.0f;

    while (state.running.load()) {
        cv::Mat frame;
        {
            std::lock_guard<std::mutex> lk(state.frameMutex);
            if (!state.latestFrame.empty()) {
                frame = state.latestFrame.clone();
            }
        }

        if (frame.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        const auto tNow = std::chrono::steady_clock::now();
        const float dt = std::chrono::duration<float>(tNow - tPrev).count();
        tPrev = tNow;
        if (dt > 0.0f) {
            fps = 0.9f * fps + 0.1f * (1.0f / dt);
        }

        const int frameW = frame.cols;
        const int frameH = frame.rows;
        cv::Mat cam = (cv::Mat_<double>(3, 3) <<
            static_cast<double>(frameW), 0.0, static_cast<double>(frameW) / 2.0,
            0.0, static_cast<double>(frameW), static_cast<double>(frameH) / 2.0,
            0.0, 0.0, 1.0
        );
        cv::Mat dist = cv::Mat::zeros(4, 1, CV_64F);

        bool phoneDetected = false;
        std::vector<std::pair<cv::Rect, float>> phones;
        {
            std::lock_guard<std::mutex> lk(state.phoneMutex);
            phoneDetected = state.phoneDetected;
            phones = state.phoneBoxes;
        }

        for (const auto& phone : phones) {
            cv::rectangle(frame, phone.first, cv::Scalar(0, 0, 255), 2);
            cv::putText(
                frame,
                cv::format("PHONE %.2f", phone.second),
                cv::Point(phone.first.x, std::max(0, phone.first.y - 8)),
                cv::FONT_HERSHEY_SIMPLEX,
                0.7,
                cv::Scalar(0, 0, 255),
                2
            );
        }

        cv::Mat resized;
        cv::resize(frame, resized, cv::Size(config::DET_W, config::DET_H));
        yunet->setInputSize(cv::Size(config::DET_W, config::DET_H));

        cv::Mat faces;
        yunet->detect(resized, faces);

        std::string dmsDir = "NO_FACE";
        std::string gaze = "UNKNOWN";
        bool hasYaw = false;
        eyeClosed = false;

        if (!faces.empty() && faces.rows > 0) {
            const float sx = static_cast<float>(frameW) / static_cast<float>(config::DET_W);
            const float sy = static_cast<float>(frameH) / static_cast<float>(config::DET_H);

            const float* f = faces.ptr<float>(0);
            const int fx = static_cast<int>(std::round(f[0] * sx));
            const int fy = static_cast<int>(std::round(f[1] * sy));
            const int fw = static_cast<int>(std::round(f[2] * sx));
            const int fh = static_cast<int>(std::round(f[3] * sy));

            const cv::Rect faceRect = clampFaceRect(fx, fy, fw, fh, frameW, frameH);
            if (!faceRect.empty()) {
                cv::rectangle(frame, faceRect, cv::Scalar(0, 255, 0), 2);
            }

            std::array<cv::Point2f, 5> kps{};
            for (int i = 0; i < 5; ++i) {
                const float px = f[5 + i * 2] * sx;
                const float py = f[5 + i * 2 + 1] * sy;
                kps[i] = cv::Point2f(px, py);
                cv::circle(frame, kps[i], 3, cv::Scalar(0, 255, 255), -1);
            }

            float yawDeg = 0.0f;
            if (solveHeadPoseYaw(kps, cam, dist, yawDeg)) {
                hasYaw = true;
                if (!yawInit) {
                    yawSmooth = yawDeg;
                    yawInit = true;
                } else {
                    yawSmooth = config::ALPHA_YAW * yawDeg + (1.0f - config::ALPHA_YAW) * yawSmooth;
                }

                cv::putText(
                    frame,
                    cv::format("Yaw: %+0.1f", yawSmooth),
                    cv::Point(20, 120),
                    cv::FONT_HERSHEY_SIMPLEX,
                    0.7,
                    cv::Scalar(255, 255, 255),
                    2
                );
            }

            const bool needEye = !hasYaw || std::fabs(yawSmooth) < 10.0f;
            if (needEye) {
                const int roiW = static_cast<int>(std::max(20.0f, fw * config::EYE_ROI_W));
                const int roiH = static_cast<int>(std::max(12.0f, fh * config::EYE_ROI_H));

                std::vector<float> xs;
                auto addEye = [&](const cv::Point2f& eyePoint) {
                    int x1 = static_cast<int>(std::round(eyePoint.x - roiW));
                    int y1 = static_cast<int>(std::round(eyePoint.y - roiH));
                    int x2 = static_cast<int>(std::round(eyePoint.x + roiW));
                    int y2 = static_cast<int>(std::round(eyePoint.y + roiH));

                    x1 = std::max(0, std::min(frameW - 1, x1));
                    y1 = std::max(0, std::min(frameH - 1, y1));
                    x2 = std::max(0, std::min(frameW, x2));
                    y2 = std::max(0, std::min(frameH, y2));
                    if (x2 <= x1 || y2 <= y1) {
                        return;
                    }

                    const float xn = pupilXNorm(frame(cv::Rect(x1, y1, x2 - x1, y2 - y1)));
                    if (xn >= 0.0f) {
                        xs.push_back(xn);
                    }
                };

                addEye(kps[0]);
                addEye(kps[1]);

                eyeClosed = xs.empty();
                if (!xs.empty()) {
                    const float gx = std::accumulate(xs.begin(), xs.end(), 0.0f) / static_cast<float>(xs.size());
                    if (!gazeInit) {
                        gazeSmooth = gx;
                        gazeInit = true;
                    } else {
                        gazeSmooth = config::ALPHA_GAZE * gx + (1.0f - config::ALPHA_GAZE) * gazeSmooth;
                    }

                    gaze = gazeFromX(gazeSmooth);
                    cv::putText(
                        frame,
                        cv::format("GazeX: %.2f (%s)", gazeSmooth, gaze.c_str()),
                        cv::Point(20, 150),
                        cv::FONT_HERSHEY_SIMPLEX,
                        0.7,
                        cv::Scalar(255, 255, 255),
                        2
                    );
                }
            }

            dmsDir = fuseHeadEye(yawSmooth, hasYaw, gaze);
        }

        if (eyeClosed) {
            eyeClosedSeconds += dt;
        } else {
            eyeClosedSeconds = 0.0f;
        }

        if (dmsDir == "LEFT" || dmsDir == "RIGHT") {
            tLookAway += dt;
        } else {
            tLookAway = 0.0f;
        }

        if (phoneDetected) {
            tPhone += dt;
        } else {
            tPhone = 0.0f;
        }

        if (dmsDir == "NO_FACE") {
            tNoFace += dt;
        } else {
            tNoFace = 0.0f;
        }

        std::string dmsState;
        if (phoneDetected) {
            dmsState = "USING_PHONE";
        } else if (eyeClosedSeconds >= config::EYE_CLOSE_STRONG) {
            dmsState = "DROWSY (EYES CLOSED)";
        } else if (eyeClosedSeconds >= config::EYE_CLOSE_TH) {
            dmsState = "WARNING: EYES CLOSED";
        } else if (dmsDir == "LEFT" || dmsDir == "RIGHT") {
            dmsState = "LOOKING_AWAY";
        } else if (dmsDir == "STRAIGHT") {
            dmsState = "NORMAL";
        } else {
            dmsState = dmsDir;
        }

        std::string warning = "NONE";
        if (tPhone >= config::TH_PHONE) {
            warning = "WARNING: USING PHONE";
        } else if (tNoFace >= config::TH_NO_FACE) {
            warning = "WARNING: NO FACE";
        } else if (tLookAway >= config::TH_LOOK_AWAY) {
            warning = "WARNING: LOOKING AWAY";
        }

        cv::putText(frame, cv::format("FPS: %.1f", fps), cv::Point(20, 40), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "DIR: " + dmsDir, cv::Point(20, 75), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 0, 255), 2);
        cv::putText(frame, "STATE: " + dmsState, cv::Point(20, 185), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 0, 255), 2);

        if (warning != "NONE") {
            cv::putText(frame, warning, cv::Point(20, 230), cv::FONT_HERSHEY_SIMPLEX, 1.1, cv::Scalar(0, 0, 255), 3);
        }

        cv::putText(
            frame,
            cv::format("EyeClosed: %s  (%.2fs)", eyeClosed ? "YES" : "NO", eyeClosedSeconds),
            cv::Point(20, 215),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(0, 255, 255),
            2
        );

        cv::putText(
            frame,
            cv::format("YOLO seq: %llu", static_cast<unsigned long long>(state.yoloSeq.load())),
            cv::Point(20, 270),
            cv::FONT_HERSHEY_SIMPLEX,
            0.7,
            cv::Scalar(255, 255, 255),
            2
        );

        cv::imshow("DMS (MT + AsyncYOLO DoubleBuffer)", frame);
        const int k = cv::waitKey(1) & 0xFF;
        if (k == 27) {
            break;
        }
    }

    state.running.store(false);
    if (tCap.joinable()) {
        tCap.join();
    }
    if (tYolo.joinable()) {
        tYolo.join();
    }

    return 0;
}
