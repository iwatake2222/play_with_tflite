// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// ----------------------------------------------------------------------
// Editor: iwatake (2020/07/24)
// ----------------------------------------------------------------------

#include <stdio.h>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <vector>

#include "ssd_anchors_calculator.h"
#include "tflite_tensors_to_detections_calculator.h"


#if defined(ANDROID) || defined(__ANDROID__)
#include <android/log.h>
#define TAG "MyApp_NDK"
#define PRINT(...) __android_log_print(ANDROID_LOG_INFO, TAG, __VA_ARGS__)
#else
#define PRINT(fmt, ...) printf("[tflite_tensors_to_detections_calculator] " fmt, __VA_ARGS__)
#endif
#define CHECK_EQ(x, y)                              \
  if (x != y) {                                                \
	PRINT("Error at %s:%d\n", __FILE__, __LINE__); \
	exit(1);                                                 \
  }

namespace mediapipe {

	int DecodeBoxes(const TfLiteTensorsToDetectionsCalculatorOptions &options_, const float* raw_boxes, const std::vector<Anchor>& anchors, std::vector<float>& boxes)
	{
		int num_boxes_ = options_.num_boxes();
		int num_coords_ = options_.num_coords();
		if (boxes.size() != num_boxes_ * num_coords_) {
			return -1;
		}
		if (anchors.size() != num_boxes_) {
			return -1;
		}

		for (int i = 0; i < num_boxes_; ++i) {
			const int box_offset = i * num_coords_ + options_.box_coord_offset();
			
			float y_center = raw_boxes[box_offset];
			float x_center = raw_boxes[box_offset + 1];
			float h = raw_boxes[box_offset + 2];
			float w = raw_boxes[box_offset + 3];
			if (options_.reverse_output_order()) {
				x_center = raw_boxes[box_offset];
				y_center = raw_boxes[box_offset + 1];
				w = raw_boxes[box_offset + 2];
				h = raw_boxes[box_offset + 3];
			}

			x_center = x_center / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
			y_center = y_center / options_.y_scale() * anchors[i].h() + anchors[i].y_center();
			if (options_.apply_exponential_on_box_size()) {
				h = std::exp(h / options_.h_scale()) * anchors[i].h();
				w = std::exp(w / options_.w_scale()) * anchors[i].w();
			} else {
				h = h / options_.h_scale() * anchors[i].h();
				w = w / options_.w_scale() * anchors[i].w();
			}

			const float ymin = y_center - h / 2.f;
			const float xmin = x_center - w / 2.f;
			const float ymax = y_center + h / 2.f;
			const float xmax = x_center + w / 2.f;

			boxes[i * num_coords_ + 0] = ymin;
			boxes[i * num_coords_ + 1] = xmin;
			boxes[i * num_coords_ + 2] = ymax;
			boxes[i * num_coords_ + 3] = xmax;

			if (options_.num_keypoints()) {
				for (int k = 0; k < options_.num_keypoints(); ++k) {
					const int offset = i * num_coords_ + options_.keypoint_coord_offset() + k * options_.num_values_per_keypoint();

					float keypoint_y = raw_boxes[offset];
					float keypoint_x = raw_boxes[offset + 1];
					if (options_.reverse_output_order()) {
						keypoint_x = raw_boxes[offset];
						keypoint_y = raw_boxes[offset + 1];
					}

					boxes[offset] = keypoint_x / options_.x_scale() * anchors[i].w() + anchors[i].x_center();
					boxes[offset + 1] = keypoint_y / options_.y_scale() * anchors[i].h() + anchors[i].y_center();
				}
			}
		}
		return 0;
	}


	int ConvertToDetections(
		const TfLiteTensorsToDetectionsCalculatorOptions &options_,
		const float* detection_boxes, const float* detection_scores,
		const int* detection_classes, std::vector<Detection>& output_detections)
	{
		for (int i = 0; i < options_.num_boxes(); ++i) {
			if (detection_scores[i] < options_.min_score_thresh()) continue;
			const int box_offset = i * options_.num_coords();
			Detection detection;
			detection.score = detection_scores[i];
			detection.class_id = detection_classes[i];
			detection.x = detection_boxes[box_offset + 1];
			detection.y = detection_boxes[box_offset + 0];
			detection.w = detection_boxes[box_offset + 3] - detection_boxes[box_offset + 1];
			detection.h = detection_boxes[box_offset + 2] - detection_boxes[box_offset + 0];
			
			// Add keypoints.
			if (options_.num_keypoints() > 0) {
				detection.keypoints.clear();
				std::vector<std::pair<int, int>> keypoints(options_.num_keypoints());
				for (int kp_id = 0; kp_id < options_.num_keypoints() * options_.num_values_per_keypoint(); kp_id += options_.num_values_per_keypoint()) {
					std::pair<float, float> keypoint;
					const int keypoint_index = box_offset + options_.keypoint_coord_offset() + kp_id;
					keypoint.first = detection_boxes[keypoint_index + 0];
					keypoint.second = detection_boxes[keypoint_index + 1];
					detection.keypoints.push_back(keypoint);
				}
			}
			output_detections.push_back(detection);
			
		}
		return 0;
	}



	int Process(const TfLiteTensorsToDetectionsCalculatorOptions &options, const float* raw_boxes, const float* raw_scores, const std::vector<Anchor>& anchors, std::vector<Detection>& output_detections) {
		std::vector<float> boxes(options.num_boxes() * options.num_coords());
		DecodeBoxes(options, raw_boxes, anchors, boxes);

		std::vector<float> detection_scores(options.num_boxes());
		std::vector<int> detection_classes(options.num_boxes());

		// Filter classes by scores.
		for (int i = 0; i < options.num_boxes(); ++i) {
			int class_id = -1;
			float max_score = -std::numeric_limits<float>::max();
			// Find the top score for box i.
			for (int score_idx = 0; score_idx < options.num_classes(); ++score_idx) {
				auto score = raw_scores[i * options.num_classes() + score_idx];
				if (options.sigmoid_score()) {
					//if (options_.has_score_clipping_thresh()) {
					//	score = score < -options_.score_clipping_thresh()
					//		? -options_.score_clipping_thresh()
					//		: score;
					//	score = score > options_.score_clipping_thresh()
					//		? options_.score_clipping_thresh()
					//		: score;
					//}
					score = 1.0f / (1.0f + std::exp(-score));
				}
				if (max_score < score) {
					max_score = score;
					class_id = score_idx;
				}
			}
			detection_scores[i] = max_score;
			detection_classes[i] = class_id;
		}

		ConvertToDetections(options, boxes.data(), detection_scores.data(), detection_classes.data(), output_detections);
		return 0;
	}

}  // namespace mediapipe
