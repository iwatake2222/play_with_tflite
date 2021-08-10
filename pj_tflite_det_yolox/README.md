# YOLOX with TensorFlow Lite / OpenCV(cv::dnn + ONNX) in C++
Sample project to run YOLOX + SORT

Click the image to open in YouTube

[![00_doc/yolox_sort.jpg](00_doc/yolox_sort.jpg)](https://youtu.be/bl19Ik4uz7c)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script (For tflite)
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano_new.sh
        - copy `saved_model_yolox_nano_480x640/model_float32.tflite` to `resource/model/yolox_nano_480x640.tflite`
    - Download the model using the following script (For onnx)
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano.sh
        - copy `saved_model_yolox_nano_480x640/yolox_nano_480x640.onnx` to `resource/model/yolox_nano_480x640.onnx`
    - Place  `resource/kite.jpg` and `resource/model/label_coco_80.txt`
        - https://user-images.githubusercontent.com/11009876/128452081-4ea8e635-5085-4d9f-b95f-cb4fb7475900.jpg
        - https://github.com/iwatake2222/play_with_tflite/files/6938693/label_coco_80.txt
    - Build  `pj_tflite_det_yolox` project (this directory)

## Notice
- By default it uses tflite model. If you want to use onnx model please change ifdef switch in `detection_engine.cpp`
    - `#define MODEL_TYPE_TFLITE`
    - `#define MODEL_TYPE_ONNX`

## Play more ?
- You can run the project on Windows, Linux (x86_64), Linux (ARM) and Android
- The project here uses a very basic model and settings
- You can try another model such as bigger input size, quantized model, etc.
    - Please modify `Model parameters` part in `detection_engine.cpp`
- You can try TensorFlow Lite with delegate
    - Please modify `Create and Initialize Inference Helper` part in `detection_engine.cpp` and cmake option
- You can try another inference engine like OpenCV, TensorRT, etc.
    - Please modify `Create and Initialize Inference Helper` part in `detection_engine.cpp` and cmake option

## Acknowledgements
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/PINTO0309/PINTO_model_zoo
