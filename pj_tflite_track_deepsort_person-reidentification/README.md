# YOLOX-Nano + Deep SORT (using person reidentification model) with TensorFlow Lite in C++

Click the image to open in YouTube. https://youtu.be/wjekIwk2cXI

[![00_doc/deepsort.jpg](00_doc/deepsort.jpg)](https://youtu.be/wjekIwk2cXI)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the YOLOX model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/132_YOLOX/download_nano_new.sh
        - copy `saved_model_yolox_nano_480x640/model_float32.tflite` to `resource/model/yolox_nano_480x640.tflite`
    - Download the Deep SORT model (person reidentification model) using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/083_Person_Reidentification/person-reidentification-retail-0300/download.sh
        - copy `model_float32.tflite` to `resource/model/person-reidentification-retail-0300.tflite`
        - Note:
            - This is the most accurate model (the largest model). You can find and use different size models
    - Build  `pj_tflite_track_deepsort_person-reidentification` project (this directory)

## Acknowledgements
- https://arxiv.org/abs/1703.07402
- https://github.com/openvinotoolkit/open_model_zoo/blob/2020.2/models/intel/person-reidentification-retail-0300/description/person-reidentification-retail-0300.md
- https://github.com/Megvii-BaseDetection/YOLOX
- https://github.com/PINTO0309/PINTO_model_zoo
