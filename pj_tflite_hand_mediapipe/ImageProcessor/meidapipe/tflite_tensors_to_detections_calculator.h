// Refecence: mediapipe\framework\formats\detection.proto
typedef struct {
	float score;
	int class_id;
	float x;
	float y;
	float w;
	float h;
	std::vector<std::pair<float, float>> keypoints;		// <x, y>
} Detection;


// Refecence:  mediapipe\graphs\hand_tracking\subgraphs\hand_detection_gpu.pbtxt
class TfLiteTensorsToDetectionsCalculatorOptions {
private:
	int _num_classes;
	int _num_boxes;
	int _num_coords;
	int _box_coord_offset;
	int _keypoint_coord_offset;
	int _num_keypoints;
	int _num_values_per_keypoint;
	int _sigmoid_score;
	float _score_clipping_thresh;
	bool _reverse_output_order;
	float _x_scale;
	float _y_scale;
	float _h_scale;
	float _w_scale;
	float _min_score_thresh;

	bool _apply_exponential_on_box_size;


public:
	TfLiteTensorsToDetectionsCalculatorOptions() {
		_num_classes = 1;
		_num_boxes = 2944;
		_num_coords = 18;			// bbox(2*2) + keypoints(7*2)
		_box_coord_offset = 0;
		_keypoint_coord_offset = 4;
		_num_keypoints = 7;
		_num_values_per_keypoint = 2;
		_sigmoid_score = true;
		_score_clipping_thresh = 100.0f;
		_reverse_output_order = true;
		_x_scale = 256.0;
		_y_scale = 256.0;
		_h_scale = 256.0;
		_w_scale = 256.0;
		_min_score_thresh = 0.7f;
		_apply_exponential_on_box_size = false;
	}
	int num_classes() const { return _num_classes; }
	int num_boxes() const { return _num_boxes; }
	int num_coords() const { return _num_coords; }
	int box_coord_offset() const { return _box_coord_offset; }
	int keypoint_coord_offset() const { return _keypoint_coord_offset; }
	int num_keypoints() const { return _num_keypoints; }
	int num_values_per_keypoint() const { return _num_values_per_keypoint; }
	int sigmoid_score() const { return _sigmoid_score; }
	float score_clipping_thresh() const { return _score_clipping_thresh; }
	bool  reverse_output_order() const { return _reverse_output_order; }
	float x_scale() const { return _x_scale; }
	float y_scale() const { return _y_scale; }
	float h_scale() const { return _h_scale; }
	float w_scale() const { return _w_scale; }
	float min_score_thresh() const { return _min_score_thresh; }
	bool  apply_exponential_on_box_size() const { return _apply_exponential_on_box_size; }
};



namespace mediapipe {
	int Process(const TfLiteTensorsToDetectionsCalculatorOptions &options, const float* raw_boxes, const float* raw_scores, const std::vector<Anchor>& anchors, std::vector<Detection>& output_detections);
}