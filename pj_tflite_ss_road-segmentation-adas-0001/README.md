# Road Segmentation Adas with TensorFlow Lite in C++

Click the image to open in YouTube. https://youtu.be/k7upHMvalWI

[![00_doc/road-segmentation-adas-0001.jpg](00_doc/road-segmentation-adas-0001.jpg)](https://youtu.be/k7upHMvalWI)

The test image data: https://motchallenge.net/vis/MOT16-14

## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/136_road-segmentation-adas-0001/download.sh
        - copy `saved_model/model_float32.tflite` to `resource/model/road-segmentation-adas-0001.tflite`
    - Build  `pj_tflite_ss_road-segmentation-adas-0001` project (this directory)

## Acknowledgements
- https://github.com/openvinotoolkit/open_model_zoo/tree/master/models/intel/road-segmentation-adas-0001
- https://github.com/PINTO0309/PINTO_model_zoo
- https://motchallenge.net/
