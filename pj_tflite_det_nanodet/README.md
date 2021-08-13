# NanoDet with ~~TensorFlow Lite~~ (cv::dnn) in C++
Sample project to run NanoDet

![00_doc/nanodet.jpg](00_doc/nanodet.jpg)

## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/072_NanoDet/download.sh
        - copy `saved_model_nanodet_320x320/model_float32.tflite` to `resource/model/nanodet_320x320.tflite`
        - copy `saved_model_nanodet_320x320/nanodet_320x320.onnx` to `resource/model/nanodet_320x320.onnx`
    - Place `resource/model/label_coco_80.txt`
        - https://github.com/iwatake2222/play_with_tflite/files/6938693/label_coco_80.txt
    - Build  `pj_tflite_det_nanodet` project (this directory)

## Note
- Currently, the project uses ONNX model and cv::dnn
    - You need OpenCV with dnn module enabled
    - I confirmed with OpenCV 4.5. Execution failed with OpenCV 4.1
- Code for tflite is WIP

## Acknowledgements
- https://github.com/RangiLyu/nanodet.git
- https://github.com/PINTO0309/PINTO_model_zoo
