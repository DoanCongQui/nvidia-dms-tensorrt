#include "yolo_worker.h"

#include "config.h"

#include <chrono>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

namespace face_ai {

void yoloThread(SharedState& state, YoloTRT& yolo) {
    uint64_t lastSeq = 0;
    int submitId = 0;
    cv::Mat local;

    while (state.running.load()) {
        uint64_t seqNow = 0;
        {
            std::lock_guard<std::mutex> lk(state.frameMutex);
            seqNow = state.frameSeq;
            if (!state.latestFrame.empty()) {
                local = state.latestFrame.clone();
            }
        }

        if (local.empty()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(2));
            continue;
        }

        if (seqNow != lastSeq) {
            lastSeq = seqNow;

            if (config::YOLO_EVERY_N <= 1 || (submitId % config::YOLO_EVERY_N == 0)) {
                const int slot = submitId & 1;
                yolo.submitAsync(slot, local);

                const int prev = slot ^ 1;
                std::vector<std::pair<cv::Rect, float>> phones;
                if (yolo.tryGetResult(prev, local, phones)) {
                    std::lock_guard<std::mutex> lk(state.phoneMutex);
                    state.phoneBoxes = std::move(phones);
                    state.phoneDetected = !state.phoneBoxes.empty();
                    state.yoloSeq.store(seqNow, std::memory_order_relaxed);
                }

                ++submitId;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

} // namespace face_ai
