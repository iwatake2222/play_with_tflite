# FastestDet with TensorFlow Lite in C++
Sample project to run FastestDet + SORT

![00_doc/demo.jpg](00_doc/demo.jpg)

https://user-images.githubusercontent.com/11009876/179514632-769158f7-d9b3-4ac8-b17b-f9693732f0a7.mp4

*FastestDet + TensorFlow Lite + Android (Pixel 4a)*

## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/308_FastestDet/download_openvino_oak_myriad_tf_tflite_tfjs_coreml.sh
        - copy `fastestdet_352x640/model_float32.tflite` to `resource/model/fastestdet_352x640.tflite`
    - Build  `pj_tflite_det_fastestdet` project (this directory)

## Acknowledgements
- https://github.com/dog-qiuqiu/FastestDet
- https://github.com/PINTO0309/PINTO_model_zoo
