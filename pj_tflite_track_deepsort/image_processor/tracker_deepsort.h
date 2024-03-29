/* Copyright 2021 iwatake2222

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#ifndef TRACKER_DEEPSORT_
#define TRACKER_DEEPSORT_

/* for general */
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <deque>
#include <list>
#include <array>
#include <memory>

/* for My modules */
#include "bounding_box.h"
#include "kalman_filter.h"


class TrackDeepSort {
private:
    static constexpr int32_t kMaxHistoryNum = 500;

public:
    typedef struct Data_ {
        BoundingBox bbox;
        BoundingBox bbox_raw;
        std::vector<float> feature;
    } Data;

public:
    TrackDeepSort(const int32_t id, const BoundingBox& bbox_det, const std::vector<float>& feature);
    ~TrackDeepSort();

    BoundingBox Predict();
    void Update(const BoundingBox& bbox_det);
    void UpdateNoDetect();

    std::deque<Data>& GetDataHistory();
    Data& GetLatestData() ;
    BoundingBox& GetLatestBoundingBox();

    const int32_t GetId() const;
    const int32_t GetUndetectedCount() const;
    const int32_t GetDetectedCount() const;

private:
    KalmanFilter CreateKalmanFilter_UniformLinearMotion(const BoundingBox& bbox_start);
    SimpleMatrix Bbox2KalmanObserved(const BoundingBox& bbox);
    SimpleMatrix Bbox2KalmanStatus(const BoundingBox& bbox);
    BoundingBox KalmanStatus2Bbox(const SimpleMatrix& X);

private:
    std::deque<Data> data_history_;
    KalmanFilter kf_;
    int32_t id_;
    int32_t cnt_detected_;
    int32_t cnt_undetected_;
};


class TrackerDeepSort {
private:
    static constexpr float kCostMax = 1.0F;

public:
    TrackerDeepSort(int32_t threshold_frame_to_delete = 2);
    ~TrackerDeepSort();
    void Reset();

    void Update(const std::vector<BoundingBox>& det_list, const std::vector<std::vector<float>>& feature_list);

    std::vector<TrackDeepSort>& GetTrackList();

private:
    float CalculateCost(TrackDeepSort& track, const BoundingBox& det_bbox, const std::vector<float>& det_feature);

private:
    std::vector<TrackDeepSort> track_list_;
    int32_t track_sequence_num_;

    int32_t threshold_frame_to_delete_;
};

#endif
