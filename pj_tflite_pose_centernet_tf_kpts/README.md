# CenterNet Object and Keypoints Detection with TensorFlow Lite in C++
Sample project to run CenterNet (OD + Keypoints) from TensorFlow 2 Detection Model Zoo

Click the image to open in YouTube. https://youtu.be/-xpBWLw-A-I

[![00_doc/centernet_kpts.jpg](00_doc/centernet_kpts.jpg)](https://youtu.be/-xpBWLw-A-I)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/042_centernet/20_tensorflow_models/download_centernet_mobilenetv2_fpn_kpts_480x640.sh
        - copy `saved_model/model_float32.tflite` to `resource/model/centernet_mobilenetv2_fpn_kpts_480x640.tflite`
    - Build  `pj_tflite_det_yolov5` project (this directory)

## Acknowledgements
- https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md
- https://tfhub.dev/tensorflow/centernet/resnet50v1_fpn_512x512_kpts/1
- https://github.com/PINTO0309/PINTO_model_zoo

