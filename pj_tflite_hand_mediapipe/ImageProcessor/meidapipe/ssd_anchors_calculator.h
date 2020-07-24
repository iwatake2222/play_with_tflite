// Refecence: mediapipe\framework\formats\object_detection\anchor.proto
class Anchor {
public:
	// Encoded anchor box center.
	float _x_center;
	float _y_center;
	// Encoded anchor box height.
	float _h;
	// Encoded anchor box width.
	float _w;

	void set_x_center(float __x_center) { _x_center = __x_center; }
	void set_y_center(float __y_center) { _y_center = __y_center; }
	void set_h(float __h) { _h = __h; }
	void set_w(float __w) { _w = __w; }
	float x_center() const { return _x_center; }
	float y_center() const { return _y_center; }
	float h() const { return _h; }
	float w() const { return _w; }
};

// Refecence: mediapipe\graphs\hand_tracking\subgraphs\hand_detection_gpu.pbtxt
class SsdAnchorsCalculatorOptions {
private:
	int _num_layers;
	float _min_scale;
	float _max_scale;
	int _input_size_height;
	int _input_size_width;
	float _anchor_offset_x;
	float _anchor_offset_y;
	int _strides_size;
	std::vector<int>_strides;
	int _aspect_ratios_size;
	std::vector<float>_aspect_ratios;
	bool _fixed_anchor_size;

	float _interpolated_scale_aspect_ratio;
	bool _reduce_boxes_in_lowest_layer;
	std::vector<int>_feature_map_width;
	int _feature_map_width_size;
	std::vector<int>_feature_map_height;
	int _feature_map_height_size;


public:
	SsdAnchorsCalculatorOptions() {
		_num_layers = 5;
		_min_scale = 0.1171875;
		_max_scale = 0.75;
		_input_size_height = 256;
		_input_size_width = 256;
		_anchor_offset_x = 0.5;
		_anchor_offset_y = 0.5;
		
		_strides.push_back(8);
		_strides.push_back(16);
		_strides.push_back(32);
		_strides.push_back(32);
		_strides.push_back(32);
		_strides_size = (int)_strides.size();
		_aspect_ratios.push_back(1.0);
		_aspect_ratios_size = (int)_aspect_ratios.size();
		_fixed_anchor_size = true;

		_interpolated_scale_aspect_ratio = 1.0;
		_reduce_boxes_in_lowest_layer = false;
		_feature_map_width.clear();
		_feature_map_width_size = (int)_feature_map_width.size();
		_feature_map_height.clear();
		_feature_map_height_size = (int)_feature_map_height.size();
	}
	int num_layers() const { return _num_layers; }
	float min_scale() const  { return _min_scale; }
	float max_scale() const  { return _max_scale; }
	int input_size_height() const  { return _input_size_height; }
	int input_size_width() const  { return _input_size_width; }
	float anchor_offset_x() const  { return _anchor_offset_x; }
	float anchor_offset_y() const  { return _anchor_offset_y; }
	int strides_size() const  { return _strides_size; }
	int strides(int index) const  { return _strides[index]; }
	int aspect_ratios_size() const  { return _aspect_ratios_size; }
	float aspect_ratios(int index) const  { return _aspect_ratios[index]; }
	bool fixed_anchor_size() const  { return _fixed_anchor_size; }

	float interpolated_scale_aspect_ratio() const { return _interpolated_scale_aspect_ratio;}
	bool reduce_boxes_in_lowest_layer() const { return _reduce_boxes_in_lowest_layer; }
	int feature_map_width_size() const { return _feature_map_width_size; }
	int feature_map_height_size() const { return _feature_map_height_size; }
	int feature_map_width(int index) const  { return _feature_map_width[index]; }
	int feature_map_height(int index) const  { return _feature_map_height[index]; }
};



namespace mediapipe {
	int GenerateAnchors(std::vector<Anchor>* anchors, const SsdAnchorsCalculatorOptions& options);
}