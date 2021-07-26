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

    kf_w_.Initialize(bbox_det.w, 1, 1, 10);
    kf_h_.Initialize(bbox_det.h, 1, 1, 10);
    kf_cx_.Initialize(GetCenterX(bbox_det), 1, 1, 5);
    kf_cy_.Initialize(GetCenterY(bbox_det), 1, 1, 5);


    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

Track::~Track()
{
}

BoundingBox Track::Predict() const
{
    BoundingBox bbox_pred = GetLatestBoundingBox();
    bbox_pred.w = kf_w_.Predict();
    bbox_pred.h = kf_h_.Predict();
    bbox_pred.x = kf_cx_.Predict() - bbox_pred.w / 2;
    bbox_pred.y = kf_cy_.Predict() - bbox_pred.h / 2;
    bbox_pred.score = 0.0F;

    return bbox_pred;
}

void Track::Update(const BoundingBox& bbox_det, bool is_detected)
{
    Data data;
    data.bbox = bbox_det;
    data.bbox_raw = bbox_det;

    data.bbox.w = kf_w_.Update(bbox_det.w);
    data.bbox.h = kf_h_.Update(bbox_det.h);
    data.bbox.x = kf_cx_.Update(GetCenterX(bbox_det)) - data.bbox.w / 2;
    data.bbox.y = kf_cy_.Update(GetCenterY(bbox_det)) - data.bbox.h / 2;

    data_history_.push_back(data);

    if (data_history_.size() > kMaxHistoryNum) {
        data_history_.pop_front();
    }

    if (is_detected) {
        cnt_detected_++;
        cnt_undetected_ = 0;
    } else {
        cnt_undetected_++;
    }
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
    for (const auto& track : track_list_) {
        bbox_pred_list.push_back(track.Predict());
    }

    /*** Assign ***/
    /* Calculate distance b/w predicted position and detected position */
    std::vector<std::vector<float>> similarity_table(track_list_.size());
    for (auto& s : similarity_table) s.assign(det_list.size(), 0);
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (int32_t i_det = 0; i_det < det_list.size(); i_det++) {
            if (bbox_pred_list[i_track].class_id == det_list[i_det].class_id) {
                similarity_table[i_track][i_det] = CalculateSimilarity(bbox_pred_list[i_track], det_list[i_det]);
            }
        }
    }

    /* Assign track and det */
    std::vector<int32_t> det_index_for_track(track_list_.size(), -1);
    std::vector<int32_t> track_index_for_det(det_list.size(), -1);
    
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        float similality_max = kThresholdIoUToTrack;
        int32_t index_det_max = -1;
        for (int32_t i_det = 0; i_det < det_list.size(); i_det++) {
            if (track_index_for_det[i_det] > 0) continue;   // already assigned
            if (similarity_table[i_track][i_det] > similality_max) {
                similality_max = similarity_table[i_track][i_det];
                index_det_max = i_det;
            }
        }

        if (index_det_max >= 0) {
            det_index_for_track[i_track] = index_det_max;
            track_index_for_det[index_det_max] = i_track;
        }
    }

#if 0
    for (int32_t i_track = 0; i_track < track_list_.size(); i_track++) {
        for (int32_t i_det = 0; i_det < det_list.size(); i_det++) {
            if (bbox_pred_list[i_track].class_id == det_list[i_det].class_id) {
                printf("%.3f  ", similarity_table[i_track][i_det]);
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
        if (assigned_det_index >= 0) {
            track_list_[i_track].Update(det_list[assigned_det_index], true);
        } else {
            track_list_[i_track].Update(bbox_pred_list[i_track], false);
        }
    }

    /*** Delete track ***/
    for (auto it = track_list_.begin(); it != track_list_.end();) {
        if (it->GetUndetectedCount() >= kThresholdCntToDelete) {
            it = track_list_.erase(it);
        } else {
            it++;
        }
    }

    /*** Add a new track ***/
    for (int32_t i = 0; i < track_index_for_det.size(); i++) {
        if (track_index_for_det[i] < 0) {
            track_list_.push_back(Track(track_sequence_num_, det_list[i]));
            track_sequence_num_++;
        }
    }
}

