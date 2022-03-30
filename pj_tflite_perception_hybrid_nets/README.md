# YOLOX with TensorFlow Lite / OpenCV(cv::dnn + ONNX) in C++
Sample project to run YOLOX + SORT

Click the image to open in YouTube. https://youtu.be/bl19Ik4uz7c

[![00_doc/yolox_sort.jpg](00_doc/yolox_sort.jpg)](https://youtu.be/bl19Ik4uz7c)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
        - copy `saved_model_yolox_nano_480x640/model_float32.tflite` to `resource/model/yolox_nano_480x640.tflite`
        - copy `saved_model_yolox_nano_480x640/yolox_nano_480x640.onnx` to `resource/model/yolox_nano_480x640.onnx`
    - Build  `pj_tflite_det_yolox` project (this directory)

## Note
- By default it uses tflite model. If you want to use onnx model please change ifdef switch in `detection_engine.cpp`
    - `#define MODEL_TYPE_TFLITE`
    - `#define MODEL_TYPE_ONNX`

## Acknowledgements
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/PINTO0309/PINTO_model_zoo
