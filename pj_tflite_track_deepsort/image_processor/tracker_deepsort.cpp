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
#include <numeric>

/* for My modules */
#include "common_helper.h"
#include "bounding_box.h"
#include "tracker_deepsort.h"
#include "hungarian_algorithm.h"


TrackDeepSort::TrackDeepSort(const int32_t id, const BoundingBox& bbox_det, const std::vector<float>& feature)
{
    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;
    data.feature = feature;
    data_history_.push_back(data);

    kf_ = CreateKalmanFilter_UniformLinearMotion(bbox_det);

    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

TrackDeepSort::~TrackDeepSort()
{
}

BoundingBox TrackDeepSort::Predict()
{
    kf_.Predict();

    BoundingBox bbox = GetLatestBoundingBox();
    BoundingBox bbox_pred = KalmanStatus2Bbox(kf_.X);   // w, y, w, h only
    bbox.w = bbox_pred.w;
    bbox.h = bbox_pred.h;
    bbox.x = bbox_pred.x;
    bbox.y = bbox_pred.y;
    bbox.score = 0.0F;

    Data data = GetLatestData();
    data.bbox = bbox;
    data.bbox_raw = bbox;
    data_history_.push_back(data);
    if (data_history_.size() > kMaxHistoryNum) {
        data_history_.pop_front();
    }

    return bbox;
}

void TrackDeepSort::Update(const BoundingBox& bbox_det)
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

void TrackDeepSort::UpdateNoDetect()
{
    cnt_undetected_++;
}

std::deque<TrackDeepSort::Data>& TrackDeepSort::GetDataHistory()
{
    return data_history_;
}

TrackDeepSort::Data& TrackDeepSort::GetLatestData()
{
    return data_history_.back();
}

BoundingBox& TrackDeepSort::GetLatestBoundingBox()
{
    return data_history_.back().bbox;
}

const int32_t TrackDeepSort::GetId() const
{
    return id_;
}

const int32_t TrackDeepSort::GetUndetectedCount() const
{
    return cnt_undetected_;
}

const int32_t TrackDeepSort::GetDetectedCount() const
{
    return cnt_detected_;
}


static constexpr int32_t kNumObserve = 4;   /* (cx, cy, area, aspect) */
static constexpr int32_t kNumStatus = 7;    /* (cx, cy, area, aspect, vx, vy, vz)   (v = speed)*/
KalmanFilter TrackDeepSort::CreateKalmanFilter_UniformLinearMotion(const BoundingBox& bbox_start)
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

SimpleMatrix TrackDeepSort::Bbox2KalmanStatus(const BoundingBox& bbox)
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

SimpleMatrix TrackDeepSort::Bbox2KalmanObserved(const BoundingBox& bbox)
{
    SimpleMatrix Z(kNumObserve, 1, {
        static_cast<double>(bbox.x + bbox.w / 2),
        static_cast<double>(bbox.y + bbox.h / 2),
        static_cast<double>(bbox.w * bbox.h),
        static_cast<double>(bbox.w) / bbox.h,
        });
    return Z;
}

BoundingBox TrackDeepSort::KalmanStatus2Bbox(const SimpleMatrix& X)
{
    BoundingBox bbox;
    bbox.w = static_cast<int32_t>(std::sqrt(X(2, 0) * X(3, 0)));
    bbox.h = static_cast<int32_t>(X(2, 0) / bbox.w);
    bbox.x = static_cast<int32_t>(X(0, 0) - bbox.w / 2);
    bbox.y = static_cast<int32_t>(X(1, 0) - bbox.h / 2);
    return bbox;
}



constexpr float TrackerDeepSort::kCostMax;  // for link error in Android Studio (clang)
TrackerDeepSort::TrackerDeepSort(int32_t threshold_frame_to_delete)
{
    track_sequence_num_ = 0;
    threshold_frame_to_delete_ = threshold_frame_to_delete;
}

TrackerDeepSort::~TrackerDeepSort()
{
}

void TrackerDeepSort::Reset()
{
    track_list_.clear();
    track_sequence_num_ = 0;
}


std::vector<TrackDeepSort>& TrackerDeepSort::GetTrackList()
{
    return track_list_;
}

static float CosineSimilarity(const std::vector<float>& feature0, const std::vector<float>& feature1)
{
    if (feature0.size() == 0 || feature1.size() == 0 || feature0.size() != feature1.size()) {
        return 999; /* invalid */
    }
    float norm_0 = 0;
    float norm_1 = 0;
    float dot = 0;
    for (size_t i = 0; i < feature0.size(); i++) {
        norm_0 += feature0[i] * feature0[i];
        norm_1 += feature1[i] * feature1[i];
        dot += feature0[i] * feature1[i];
    }
    
    if (norm_0 == 0 || norm_1 == 0) return 999; /* invalid */

    return (std::max)(0.0f, dot / (std::sqrt(norm_0) * std::sqrt(norm_1)));
}

//static float EuclidDistance(const std::array<float, 512>& feature0, const std::array<float, 512>& feature1)
//{
//    float distance = 0;
//    for (int32_t i = 0; i < feature0.size(); i++) {
//        distance += (feature0[i] - feature1[i]) * (feature0[i] - feature1[i]);
//    }
//    return std::sqrtf(distance);
//}

static inline float AdjustFeatureSimilarity(float value)
{
    /* experimentally determined */
    /* similarity becomes around 0.8 - 0.99 even b/w different individuals */
    static constexpr float kScake = 10;
    static constexpr float kOffset = 1.0f - 1 / kScake;
    value = (value - kOffset) * kScake;
    return (std::max)(0.0f, value);
}

float TrackerDeepSort::CalculateCost(TrackDeepSort& track, const BoundingBox& det_bbox, const std::vector<float>& det_feature)
{
    const auto& track_bbox = track.GetLatestBoundingBox();

    /***  Shouldn't match far object ***/
    const double distance_image_pow2 = std::pow(track_bbox.x - det_bbox.x, 2) + std::pow(track_bbox.y - det_bbox.y, 2);
    const double threshold_distance = std::pow((track_bbox.w + track_bbox.h + det_bbox.w + det_bbox.h) / 4, 2) * 4; /* experimentally determined */
    if (distance_image_pow2 > threshold_distance) {
        return kCostMax;
    }

    /*** Calculate IOU ***/
    constexpr float weight_iou = 1.0f;
    float iou = BoundingBoxUtils::CalculateIoU(track_bbox, det_bbox);

    /*** check class id ***/
    /* those two objects are difference if those of class id are difference */
    /* however, if iou is big enough, they can be the same (detector may output wrong class id) */
    if ((iou < 0.8) && (track_bbox.class_id != det_bbox.class_id)) {
        return kCostMax;
    }

    /*** Calculate cosine similarity of feature (DEEP) ***/
    float weight_feature = 1.0f;
    float similarity_feature = 0;

    /* compare "the feature of the det object at the current frame" with "the features in the past frames of the tracked object"  */
    /* just comparaing with the previous frame may not be enough. so I compare with those in the past few frames. but no need to compare every frame. maybe once every 5 frames */
    std::vector<float> similarity_history;
    for (int32_t i = static_cast<int32_t>(track.GetDataHistory().size()) - 2; i >= 0; i -= 5) {
        const auto& data = track.GetDataHistory()[i];
        if (data.bbox_raw.score == 0) continue; /* do not compare if the object was not detected */
        float val = CosineSimilarity(data.feature, det_feature);    /* 0.0(different) - 1.0(same) */
        if (val == 999) {
            weight_feature = 0.0f;  /* do not use appearance feature if it's invalid (objects whose feature is not calculated) */
            break;
        }
        similarity_history.push_back(val);
        if (similarity_history.size() > 10) break;  /* use the feature up to past 50 (5 * 10) frame */
    }
    if (similarity_history.size() > 0) {
        similarity_feature = std::accumulate(similarity_history.begin(), similarity_history.end(), 0.0f) / similarity_history.size();   /* take average similarity */
    }

    similarity_feature = AdjustFeatureSimilarity(similarity_feature);

    /* Calculate similarity using some metrcs */
    float similarity = (weight_feature * similarity_feature + weight_iou * iou) / (weight_feature + weight_iou);

    return kCostMax - similarity;
}


void TrackerDeepSort::Update(const std::vector<BoundingBox>& det_list, const std::vector<std::vector<float>>& feature_list)
{
    /*** Predict the position at the current frame using the previous status for all tracked bbox ***/
    for (auto& track : track_list_) {
        track.Predict();
    }

    /*** Association ***/
    /* Calculate IoU b/w predicted position and detected position */
    size_t size_cost_matrix = (std::max)(track_list_.size(), det_list.size());  /* workaround: my hungarian algorithm sometimes outputs wrong result when the input matrix is not squared */
    std::vector<std::vector<float>> cost_matrix(size_cost_matrix, std::vector<float>(size_cost_matrix, kCostMax));
    for (size_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (size_t i_det = 0; i_det < det_list.size(); i_det++) {
            cost_matrix[i_track][i_det] = CalculateCost(track_list_[i_track], det_list[i_det], feature_list[i_det]);
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
            track_list_[i_track].GetLatestData().feature = feature_list[assigned_det_index];
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
            track_list_.push_back(TrackDeepSort(track_sequence_num_, det_list[i], feature_list[i]));
            track_sequence_num_++;
        }
    }
}

