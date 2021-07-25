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
#ifndef TRACKER_
#define TRACKER_

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
#include "inference_helper.h"
#include "bounding_box.h"

class KalmanFilter {
public:
    KalmanFilter() {}
    ~KalmanFilter() {}
    void Initialize(int32_t start_value, float start_deviation, float deviation_true, float deviation_noise);
    int32_t Update(int32_t observation_value);

private:
    float x_prev_;
    float P_prev_;
    float K_;
    float P_;
    float x_;

    float start_deviation_;
    float deviation_true_;
    float deviation_noise_;
};

class Track {
public:
    typedef struct Data_ {
        BoundingBox bbox;
        BoundingBox bbox_raw;
    } Data;

public:
    Track(const int32_t id, const BoundingBox& bbox);
    ~Track();

    void PreUpdate();
    void Update(const BoundingBox& bbox);
    void UpdateNoDet();

    Data& Track::GetLatestData();
    BoundingBox& GetLatestBoundingBox();
    std::deque<Data>& GetTrackHistory();

    int32_t GetUndetectedCount() const;

public:
    std::deque<Data> data_history_;
    KalmanFilter kf_cx;
    KalmanFilter kf_cy;
    KalmanFilter kf_w;
    KalmanFilter kf_h;
    int32_t id_;
    int32_t cnt_detected_;
    int32_t cnt_undetected_;
};

class Tracker {
public:
    Tracker();
    ~Tracker();
    void Reset();

    void Update(const std::vector<BoundingBox>& det_list);

    std::list<Track>& GetTrackList();

private:


private:
    std::list<Track> track_list_;
    int32_t track_id_;
};

#endif
