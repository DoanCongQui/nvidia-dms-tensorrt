#include "nms.h"

#include <algorithm>
#include <numeric>

namespace face_ai {

float iou(const cv::Rect& a, const cv::Rect& b) {
    const int inter = (a & b).area();
    const int uni = a.area() + b.area() - inter;
    return uni > 0 ? static_cast<float>(inter) / static_cast<float>(uni) : 0.0f;
}

std::vector<int> nmsIndices(const std::vector<cv::Rect>& boxes, const std::vector<float>& scores, float nmsTh) {
    std::vector<int> order(boxes.size());
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int i, int j) {
        return scores[i] > scores[j];
    });

    std::vector<int> keep;
    std::vector<char> removed(boxes.size(), 0);

    for (size_t oi = 0; oi < order.size(); ++oi) {
        const int i = order[oi];
        if (removed[i]) {
            continue;
        }

        keep.push_back(i);
        for (size_t oj = oi + 1; oj < order.size(); ++oj) {
            const int j = order[oj];
            if (!removed[j] && iou(boxes[i], boxes[j]) > nmsTh) {
                removed[j] = 1;
            }
        }
    }

    return keep;
}

} // namespace face_ai
