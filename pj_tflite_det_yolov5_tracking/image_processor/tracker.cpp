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

Track::Track(const int32_t id, const BoundingBox& bbox)
{
    Data data;
    data.bbox = bbox;
    data.bbox_raw = bbox;
    data.is_detected = true;
    data_history_.push_back(data);

    cnt_detected_ = 1;
    cnt_undetected_ = 0;
    id_ = id;
}

Track::~Track()
{

}



void Track::PreUpdate()
{
    Data data;
    data.bbox = GetLatestBoundingBox();
    data.bbox_raw = GetLatestBoundingBox();
    data.is_detected = false;
    data_history_.push_back(data);
}

void Track::Update(const BoundingBox& bbox)
{
    data_history_.back().bbox = bbox;
    data_history_.back().bbox_raw = bbox;

    cnt_detected_++;
    cnt_undetected_ = 0;
}

void Track::UpdateNoDet()
{
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

float CalculateIoU(const BoundingBox& obj0, const BoundingBox& obj1)
{
    int32_t interx0 = (std::max)(obj0.x, obj1.x);
    int32_t intery0 = (std::max)(obj0.y, obj1.y);
    int32_t interx1 = (std::min)(obj0.x + obj0.w, obj1.x + obj1.w);
    int32_t intery1 = (std::min)(obj0.y + obj0.h, obj1.y + obj1.h);
    if (interx1 < interx0 || intery1 < intery0) return 0;

    int32_t area0 = obj0.w * obj0.h;
    int32_t area1 = obj1.w * obj1.h;
    int32_t areaInter = (interx1 - interx0) * (intery1 - intery0);
    int32_t areaSum = area0 + area1 - areaInter;

    return static_cast<float>(areaInter) / areaSum;
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
                float similarity = CalculateIoU(track_bbox, det_list[det_index]);
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
        int32_t index_max = -1;
        for (int32_t det_index = 0; det_index < det_list.size(); det_index++) {
            if (similarity_table[track_index][det_index] > similality_max) {
                similality_max = similarity_table[track_index][det_index];
                index_max = det_index;
            }
        }
        if (index_max >= 0) {
            det_index_for_track[track_index] = index_max;
            track_index_for_det[index_max] = track_index;
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
        if (track.GetUndetectedCount() > 2) {
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


