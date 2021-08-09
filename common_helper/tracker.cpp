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


Track::Track(const int32_t id, const BoundingBox& bbox_det)
{
    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;
    data_history_.push_back(data);

    kf_ = CreateKalmanFilter_UniformLinearMotion(bbox_det);

    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

Track::~Track()
{
}

BoundingBox Track::Predict()
{
    kf_.Predict();

    BoundingBox bbox = GetLatestBoundingBox();
    BoundingBox bbox_pred = KalmanStatus2Bbox(kf_.X);   // w, y, w, h only
    bbox.w = bbox_pred.w;
    bbox.h = bbox_pred.h;
    bbox.x = bbox_pred.x;
    bbox.y = bbox_pred.y;
    bbox.score = 0.0F;

    Data data;
    data.bbox = bbox;
    data.bbox_raw = bbox;
    data_history_.push_back(data);
    if (data_history_.size() > kMaxHistoryNum) {
        data_history_.pop_front();
    }

    return bbox;
}

void Track::Update(const BoundingBox& bbox_det)
{
    kf_.Update(Bbox2KalmanObserved(bbox_det));

    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;

    BoundingBox& bbox = data_history_.back().bbox;
    BoundingBox& bbox_raw = data_history_.back().bbox_raw;
    BoundingBox bbox_est = KalmanStatus2Bbox(kf_.X);   // w, y, w, h only
    bbox_raw = bbox_det;
    bbox = bbox_det;
    bbox.w = bbox_est.w;
    bbox.h = bbox_est.h;
    bbox.x = bbox_est.x;
    bbox.y = bbox_est.y;
    
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


static constexpr int32_t kNumObserve = 4;   /* (cx, cy, area, aspect) */
static constexpr int32_t kNumStatus = 7;    /* (cx, cy, area, aspect, vx, vy, vz)   (v = speed)*/
KalmanFilter Track::CreateKalmanFilter_UniformLinearMotion(const BoundingBox& bbox_start)
{
    /*** X(t) = F * X(t-1) + w(t) ***/
    /* Matrix to calculate X(t) from X(t-1). assume uniform motion: x(t) = x(t-1) + vt, v(t) = v(t-1) */
    const SimpleMatrix F(kNumStatus, kNumStatus, {
        1, 0, 0, 0, 1, 0, 0,
        0, 1, 0, 0, 0, 1, 0,
        0, 0, 1, 0, 0, 0, 1,
        0, 0, 0, 1, 0, 0, 0,
        0, 0, 0, 0, 1, 0, 0,
        0, 0, 0, 0, 0, 1, 0,
        0, 0, 0, 0, 0, 0, 1,
        });


    /* w(t), = noise, follows Q */
    const SimpleMatrix Q(kNumStatus, kNumStatus, {
        1, 0, 0, 0,    0,    0,     0,
        0, 1, 0, 0,    0,    0,     0,
        0, 0, 1, 0,    0,    0,     0,
        0, 0, 0, 1,    0,    0,     0,
        0, 0, 0, 0, 0.01,    0,     0,
        0, 0, 0, 0,    0, 0.01,     0,
        0, 0, 0, 0,    0,    0, 0.001,
        });

    /*** Z(t) = H * X(t) + v(t) ***/
    /* Matrix to calculate Z(observed value) from X(internal status) */
    const SimpleMatrix H(kNumObserve, kNumStatus, {
        1, 0, 0, 0, 0, 0, 0,
        0, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 0, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0,
        });

    /* v(t), = noise, follows R */
    const SimpleMatrix R(kNumObserve, kNumObserve, {
        1, 0,  0,  0,
        0, 1,  0,  0,
        0, 0, 10,  0,
        0, 0,  0, 10,
        });

    /* First internal status */
    SimpleMatrix P0 = SimpleMatrix::IdentityMatrix(kNumStatus);
    P0 = P0 * 10;   /* Set big noise at first to make K=1 and trust observed value rather than estimated value */

    const SimpleMatrix X0 = Bbox2KalmanStatus(bbox_start);

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

SimpleMatrix Track::Bbox2KalmanStatus(const BoundingBox& bbox)
{
    SimpleMatrix X(kNumStatus, 1, {
        static_cast<double>(bbox.x + bbox.w / 2),
        static_cast<double>(bbox.y + bbox.h / 2),
        static_cast<double>(bbox.w * bbox.h),
        static_cast<double>(bbox.w) / bbox.h,
        0,
        0,
        0
        });
    return X;
}

SimpleMatrix Track::Bbox2KalmanObserved(const BoundingBox& bbox)
{
    SimpleMatrix Z(kNumObserve, 1, {
        static_cast<double>(bbox.x + bbox.w / 2),
        static_cast<double>(bbox.y + bbox.h / 2),
        static_cast<double>(bbox.w * bbox.h),
        static_cast<double>(bbox.w) / bbox.h,
        });
    return Z;
}

BoundingBox Track::KalmanStatus2Bbox(const SimpleMatrix& X)
{
    BoundingBox bbox;
    bbox.w = static_cast<int32_t>(std::sqrt(X(2, 0) * X(3, 0)));
    bbox.h = static_cast<int32_t>(X(2, 0) / bbox.w);
    bbox.x = static_cast<int32_t>(X(0, 0) - bbox.w / 2);
    bbox.y = static_cast<int32_t>(X(1, 0) - bbox.h / 2);
    return bbox;
}



constexpr float Tracker::kCostMax;  // for link error in Android Studio (clang)
Tracker::Tracker()
{
    track_sequence_num_ = 0;
    threshold_frame_to_delete_ = 2;
    threshold_iou_to_track_ = 0.3F;
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
    float iou = BoundingBoxUtils::CalculateIoU(bbox0, bbox1);
    if (iou > 0.9) {
        /* must be the same object (do not check class id because class id may be mistaken) */
    } else if (iou < threshold_iou_to_track_) {
        /* cannot be the same object */
        iou = 0;
    } else {
        if (bbox0.class_id == bbox1.class_id) {
            /* can be the same object */
        } else {
            /* cannot be the same object */
            iou = 0;
        }
    }

    return kCostMax - iou;
}

void Tracker::Update(const std::vector<BoundingBox>& det_list)
{
    /*** Predict the position at the current frame using the previous status for all tracked bbox ***/
    std::vector<BoundingBox> bbox_pred_list;
    for (auto& track : track_list_) {
        BoundingBox bbox_prd = track.Predict();
        bbox_pred_list.push_back(bbox_prd);
    }

    /*** Association ***/
    /* Calculate IoU b/w predicted position and detected position */
    size_t size_cost_matrix = (std::max)(track_list_.size(), det_list.size());  /* workaround: my hungarian algorithm sometimes outputs wrong result when the input matrix is not squared */
    std::vector<std::vector<float>> cost_matrix(size_cost_matrix, std::vector<float>(size_cost_matrix, kCostMax));
    for (size_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (size_t i_det = 0; i_det < det_list.size(); i_det++) {
            cost_matrix[i_track][i_det] = CalculateSimilarity(bbox_pred_list[i_track], det_list[i_det]);
        }
    }

    /* Assign track and det */
    std::vector<int32_t> det_index_for_track(size_cost_matrix, -1);
    std::vector<int32_t> track_index_for_det(size_cost_matrix, -1);
    if (track_list_.size() > 0 && det_list.size() > 0) {
        HungarianAlgorithm<float> solver(cost_matrix);
        solver.Solve(det_index_for_track, track_index_for_det);
    }

#if 0
    for (size_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (size_t i_det = 0; i_det < det_list.size(); i_det++) {
            printf("%.3f  ", cost_matrix[i_track][i_det]);
        }
        printf("\n");
    }

    printf("track:  det\n");
    for (size_t i = 0; i < det_index_for_track.size(); i++) {
        printf("%3d:  %3d\n", i, det_index_for_track[i]);
    }
    printf("det:  track\n");
    for (size_t i = 0; i < track_index_for_det.size(); i++) {
        printf("%3d:  %3d\n", i, track_index_for_det[i]);
    }
#endif

    /*** Update track ***/
    std::vector<bool> is_det_assigned_list(size_cost_matrix, false);
    for (size_t i_track = 0; i_track < track_list_.size(); i_track++) {
        int32_t assigned_det_index = det_index_for_track[i_track];
        if (assigned_det_index >= 0 && assigned_det_index < static_cast<int32_t>(det_list.size()) && cost_matrix[i_track][assigned_det_index] < kCostMax) {
            track_list_[i_track].Update(det_list[assigned_det_index]);
            is_det_assigned_list[assigned_det_index] = true;
        } else{
            track_list_[i_track].UpdateNoDetect();
        }
    }

    /*** Delete tracks ***/
    for (auto it = track_list_.begin(); it != track_list_.end();) {
        if (it->GetUndetectedCount() >= threshold_frame_to_delete_) {
            it = track_list_.erase(it);
        } else {
            it++;
        }
    }

    /*** Add new tracks ***/
    for (size_t i = 0; i < det_list.size(); i++) {
        if (is_det_assigned_list[i] == false) {
            track_list_.push_back(Track(track_sequence_num_, det_list[i]));
            track_sequence_num_++;
        }
    }
}

