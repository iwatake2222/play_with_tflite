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

void KalmanFilter::Initialize(int32_t start_value, float start_deviation, float deviation_true, float deviation_noise)
{
    start_deviation_ = start_deviation;
    deviation_true_ = deviation_true;
    deviation_noise_ = deviation_noise;

    x_prev_ = static_cast<float>(start_value);
    P_prev_ = start_deviation_;
    K_ = P_prev_ / (P_prev_ + deviation_noise_);
    P_ = deviation_noise_ * P_prev_ / (P_prev_ + deviation_noise_);
    x_ = x_prev_ + K_ * (start_value - x_prev_);
}

int32_t KalmanFilter::Update(int32_t observation_value)
{
    x_prev_ = x_;
    P_prev_ = P_ + deviation_true_;
    K_ = P_prev_ / (P_prev_ + deviation_noise_);
    x_ = x_prev_ + K_ * (observation_value - x_prev_);
    P_ = deviation_noise_ * P_prev_ / (P_prev_ + deviation_noise_);

    return static_cast<int32_t>(x_);
}



Track::Track(const int32_t id, const BoundingBox& bbox)
{
    Data data;
    data.bbox = bbox;
    data.bbox_raw = bbox;
    data_history_.push_back(data);

    kf_w.Initialize(bbox.w, 1, 1, 10);
    kf_h.Initialize(bbox.h, 1, 1, 10);
    kf_cx.Initialize(GetCenterX(bbox), 1, 1, 10);
    kf_cy.Initialize(GetCenterY(bbox), 1, 1, 10);


    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

Track::~Track()
{

}



void Track::PreUpdate()
{
    auto& previous_bbox = GetLatestBoundingBox();
    Data data;
    data.bbox = previous_bbox;
    data.bbox_raw = previous_bbox;

    data.bbox.w = kf_w.Update(previous_bbox.w);
    data.bbox.h = kf_h.Update(previous_bbox.h);
    data.bbox.x = kf_cx.Update(GetCenterX(previous_bbox)) - data.bbox.w / 2;
    data.bbox.y = kf_cy.Update(GetCenterY(previous_bbox)) - data.bbox.h / 2;

    data_history_.push_back(data);

    if (data_history_.size() > 400) {
        data_history_.pop_front();
    }
}

void Track::Update(const BoundingBox& bbox)
{
    auto& latest_track_data = data_history_.back();
    latest_track_data.bbox = bbox;
    latest_track_data.bbox_raw = bbox;

    latest_track_data.bbox.w = kf_w.Update(bbox.w);
    latest_track_data.bbox.h = kf_h.Update(bbox.h);
    latest_track_data.bbox.x = kf_cx.Update(GetCenterX(bbox)) - latest_track_data.bbox.w / 2;
    latest_track_data.bbox.y = kf_cy.Update(GetCenterY(bbox)) - latest_track_data.bbox.h / 2;

    cnt_detected_++;
    cnt_undetected_ = 0;
}

void Track::UpdateNoDet()
{
    auto& latest_track_data = data_history_.back();
    latest_track_data.bbox.score = 0.0F;
    latest_track_data.bbox_raw.score = 0.0F;

    cnt_undetected_++;
}

int32_t Track::GetUndetectedCount() const 
{
    return cnt_undetected_;
}

BoundingBox& Track::GetLatestBoundingBox()
{
    return data_history_.back().bbox;
}


Track::Data& Track::GetLatestData()
{
    return data_history_.back();
}

std::deque<Track::Data>& Track::GetTrackHistory()
{
    return data_history_;
}


Tracker::Tracker()
{
    track_id_ = 0;
}

Tracker::~Tracker()
{
}

void Tracker::Reset()
{
    track_list_.clear();
    track_id_ = 0;
}


std::list<Track>& Tracker::GetTrackList()
{
    return track_list_;
}


void Tracker::Update(const std::vector<BoundingBox>& det_list)
{
    for (auto& track : track_list_) {
        track.PreUpdate();
    }
    
    
    /*** assign ***/
    /* Calculate distance b/w tracked objects and detected objects */
    std::vector<std::vector<float>> similarity_table(track_list_.size());
    for (auto& s : similarity_table) s.assign(det_list.size(), 0);
    int32_t track_index = 0;
    for (auto& track : track_list_) {
        const auto& track_bbox = track.GetLatestBoundingBox();
        for (int32_t det_index = 0; det_index < det_list.size(); det_index++) {
            if (track_bbox.class_id == det_list[det_index].class_id) {
                float similarity = BoundingBoxUtils::CalculateIoU(track_bbox, det_list[det_index]);
                similarity_table[track_index][det_index] = similarity;
            }
        }
        track_index++;
    }
    
    std::vector<int32_t> det_index_for_track(track_list_.size(), -1);
    std::vector<int32_t> track_index_for_det(det_list.size(), -1);
    track_index = 0;
    for (auto& track : track_list_) {
        float similality_max = 0.5;
        int32_t index_det_max = -1;
        for (int32_t det_index = 0; det_index < det_list.size(); det_index++) {
            if (similarity_table[track_index][det_index] > similality_max) {
                similality_max = similarity_table[track_index][det_index];
                index_det_max = det_index;
            }
        }

        if (index_det_max >= 0) {
            det_index_for_track[track_index] = index_det_max;
            track_index_for_det[index_det_max] = track_index;
        }
        track_index++;
    }

#if 0
    track_index = 0;
    for (auto& track : track_list_) {
        for (int32_t det_index = 0; det_index < det_list.size(); det_index++) {
            printf("%.3f  ", similarity_table[track_index][det_index]);
        }
        printf("\n");
        track_index++;
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


    track_index = 0;
    for (auto& track : track_list_) {
        int32_t assigned_det_index = det_index_for_track[track_index];
        if (assigned_det_index >= 0) {
            track.Update(det_list[assigned_det_index]);
        } else {
            track.UpdateNoDet();
        }
        track_index++;
    }

    track_list_.remove_if([](Track& track) {
        if (track.GetUndetectedCount() >= 2) {
            return true;
        } else {
            return false;
        }
    });


    int32_t det_index = 0;
    for (auto& det : track_index_for_det) {
        if (det < 0) {
            track_list_.push_back(Track(track_id_, det_list[det_index]));
            track_id_++;
        }
        det_index++;
    }
}


