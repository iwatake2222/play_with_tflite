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
#include "common_helper.h"
#include "bounding_box.h"
#include "tracker.h"
#include "hungarian_algorithm.h"

static int32_t GetCenterX(const BoundingBox& bbox)
{
    return bbox.x + bbox.w / 2;
}

static int32_t GetCenterY(const BoundingBox& bbox)
{
    return bbox.y + bbox.h / 2;
}


Track::Track(const int32_t id, const BoundingBox& bbox_det)
{
    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;
    data_history_.push_back(data);

    kf_w_ = CreateKalmanFilter_UniformLinearMotion(bbox_det.w);
    kf_h_ = CreateKalmanFilter_UniformLinearMotion(bbox_det.h);
    kf_cx_ = CreateKalmanFilter_UniformLinearMotion(GetCenterX(bbox_det));
    kf_cy_ = CreateKalmanFilter_UniformLinearMotion(GetCenterY(bbox_det));

    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

Track::~Track()
{
}

BoundingBox Track::Predict()
{
    kf_w_.Predict();
    kf_h_.Predict();
    kf_cx_.Predict();
    kf_cy_.Predict();

    BoundingBox bbox_pred = GetLatestBoundingBox();
    bbox_pred.w = static_cast<int32_t>(kf_w_.X(0, 0));
    bbox_pred.h = static_cast<int32_t>(kf_h_.X(0, 0));
    bbox_pred.x = static_cast<int32_t>(kf_cx_.X(0, 0) - bbox_pred.w / 2);
    bbox_pred.y = static_cast<int32_t>(kf_cy_.X(0, 0) - bbox_pred.h / 2);
    bbox_pred.score = 0.0F;

    Data data;
    data.bbox = bbox_pred;
    data.bbox_raw = bbox_pred;
    data_history_.push_back(data);
    if (data_history_.size() > kMaxHistoryNum) {
        data_history_.pop_front();
    }

    return bbox_pred;
}

void Track::Update(const BoundingBox& bbox_det)
{
    kf_w_.Update({ 1, 1, { static_cast<double>(bbox_det.w) } });
    kf_h_.Update({ 1, 1, { static_cast<double>(bbox_det.h) } });
    kf_cx_.Update({ 1, 1, { static_cast<double>(GetCenterX(bbox_det)) } });
    kf_cy_.Update({ 1, 1, { static_cast<double>(GetCenterY(bbox_det)) } });

    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;

    BoundingBox& bbox = data_history_.back().bbox;
    BoundingBox& bbox_raw = data_history_.back().bbox_raw;

    bbox.w = static_cast<int32_t>(kf_w_.X(0, 0));
    bbox.h = static_cast<int32_t>(kf_h_.X(0, 0));
    bbox.x = static_cast<int32_t>(kf_cx_.X(0, 0) - bbox.w / 2);
    bbox.y = static_cast<int32_t>(kf_cy_.X(0, 0) - bbox.h / 2);
    bbox.score = bbox_det.score;
    bbox_raw = bbox_det;
    
    cnt_detected_++;
    cnt_undetected_ = 0;
}

void Track::UpdateNoDetect()
{
    cnt_undetected_++;
}

std::deque<Track::Data>& Track::GetDataHistory()
{
    return data_history_;
}

const Track::Data& Track::GetLatestData() const
{
    return data_history_.back();
}

const BoundingBox& Track::GetLatestBoundingBox() const
{
    return data_history_.back().bbox;
}

const int32_t Track::GetId() const
{
    return id_;
}

const int32_t Track::GetUndetectedCount() const
{
    return cnt_undetected_;
}

const int32_t Track::GetDetectedCount() const
{
    return cnt_detected_;
}

KalmanFilter Track::CreateKalmanFilter_UniformLinearMotion(int32_t start_value)
{
    static constexpr int32_t kNumObserve = 1;	/* (x) */
    static constexpr int32_t kNumStatus = 2;	/* (x, v) */
    static constexpr double delta_t = 0.5;
    static constexpr double sigma_true = 0.3;
    static constexpr double sigma_observe = 0.1;

    /*** X(t) = F * X(t-1) + w(t) ***/
    /* Matrix to calculate X(t) from X(t-1). assume uniform motion: x(t) = x(t-1) + vt, v(t) = v(t-1) */
    const SimpleMatrix F(kNumStatus, kNumStatus, {
        1, delta_t,
        0, 1
        });

    /* Matrix to calculate delta_X from noise(=w). Assume w as accel */
    const SimpleMatrix G(kNumStatus, 1, {
        delta_t * delta_t / 2,
        delta_t
        });

    /* w(t), = noise, follows Q */
    const SimpleMatrix Q = G * G.Transpose() * (sigma_true * sigma_true);

    /*** Z(t) = H * X(t) + v(t) ***/
    /* Matrix to calculate Z(observed value) from X(internal status) */
    const SimpleMatrix H(kNumObserve, kNumStatus, {
        1, 0
        });

    /* v(t), = noise, follows R */
    const SimpleMatrix R(1, 1, { sigma_observe * sigma_observe });

    /* First internal status */
    const SimpleMatrix P0(kNumStatus, kNumStatus, {
        0, 0,
        0, 0
        });

    const SimpleMatrix X0(kNumStatus, 1, {
        static_cast<double>(start_value),
        0
        });

    KalmanFilter kf;
    kf.Initialize(
        F,
        Q,
        H,
        R,
        X0,
        P0
    );

    return kf;
}





Tracker::Tracker()
{
    track_sequence_num_ = 0;
}

Tracker::~Tracker()
{
}

void Tracker::Reset()
{
    track_list_.clear();
    track_sequence_num_ = 0;
}


std::vector<Track>& Tracker::GetTrackList()
{
    return track_list_;
}

float Tracker::CalculateSimilarity(const BoundingBox& bbox0, const BoundingBox& bbox1)
{
    float similarity = BoundingBoxUtils::CalculateIoU(bbox0, bbox1);
    return similarity;
}

void Tracker::Update(const std::vector<BoundingBox>& det_list)
{
    /*** Predict the position at the current frame using the previous status for all tracked bbox ***/
    std::vector<BoundingBox> bbox_pred_list;
    for (auto& track : track_list_) {
        BoundingBox bbox_prd = track.Predict();
        bbox_pred_list.push_back(bbox_prd);
    }

    /*** Assign ***/
    /* Calculate IoU b/w predicted position and detected position */
    std::vector<std::vector<float>> cost_matrix(track_list_.size(), std::vector<float>(det_list.size(), kCostMax));
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (int32_t i_det = 0; i_det < det_list.size(); i_det++) {
            if (bbox_pred_list[i_track].class_id == det_list[i_det].class_id) {
                cost_matrix[i_track][i_det] = kCostMax - CalculateSimilarity(bbox_pred_list[i_track], det_list[i_det]);
            }
        }
    }

    /* Assign track and det */
    std::vector<int32_t> det_index_for_track(track_list_.size(), -1);
    std::vector<int32_t> track_index_for_det(det_list.size(), -1);
    if (track_list_.size() > 0 && det_list.size() > 0) {
        HungarianAlgorithm<float> solver(cost_matrix);
        solver.Solve(det_index_for_track, track_index_for_det);
    }

#if 0
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (int32_t i_det = 0; i_det < det_list.size(); i_det++) {
            if (bbox_pred_list[i_track].class_id == det_list[i_det].class_id) {
                printf("%.3f  ", cost_matrix[i_track][i_det]);
            }
        }
        printf("\n");
    }

    printf("track:  det\n");
    for (int32_t i = 0; i < det_index_for_track.size(); i++) {
        printf("%3d:  %3d\n", i, det_index_for_track[i]);
    }
    printf("det:  track\n");
    for (int32_t i = 0; i < track_index_for_det.size(); i++) {
        printf("%3d:  %3d\n", i, track_index_for_det[i]);
    }
#endif

    /*** Update track ***/
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        int32_t assigned_det_index = det_index_for_track[i_track];
        if (assigned_det_index >= 0 && assigned_det_index < det_list.size() && cost_matrix[i_track][assigned_det_index] < kCostMax) {
            track_list_[i_track].Update(det_list[assigned_det_index]);
        } else{
            track_list_[i_track].UpdateNoDetect();
        }
    }

    /*** Delete tracks ***/
    for (auto it = track_list_.begin(); it != track_list_.end();) {
        if (it->GetUndetectedCount() >= kThresholdCntToDelete) {
            it = track_list_.erase(it);
        } else {
            it++;
        }
    }

    /*** Add new tracks ***/
    for (int32_t i = 0; i < track_index_for_det.size(); i++) {
        if (track_index_for_det[i] < 0) {
            track_list_.push_back(Track(track_sequence_num_, det_list[i]));
            track_sequence_num_++;
        }
    }
}

