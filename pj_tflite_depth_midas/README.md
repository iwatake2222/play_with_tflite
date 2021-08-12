# Depth estimation (midas/v2_1_small) with TensorFlow Lite in C++
Sample project to run depth estimation

Click the image to open in YouTube. https://youtu.be/ho6KGlHmWHk

[![00_doc/depth_midas.jpg](00_doc/depth_midas.jpg)](https://youtu.be/ho6KGlHmWHk)


## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model from TensorFlow Hub:
        - https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1
        - copy to `resource/model/lite-model_midas_v2_1_small_1_lite_1.tflite`
    - Build  `pj_tflite_depth_midas` project (this directory)

## Play more ?
- You can run the project on Windows, Linux (x86_64), Linux (ARM) and Android
- The project here uses a very basic model and settings
- You can try another model such as bigger input size, quantized model, etc.
    - Please modify `Model parameters` part in `depth_engine.cpp`
- You can try TensorFlow Lite with delegate
    - Please modify `Create and Initialize Inference Helper` part in `depth_engine.cpp` and cmake option
- You can try another inference engine like OpenCV, TensorRT, etc.
    - Please modify `Create and Initialize Inference Helper` part in `depth_engine.cpp` and cmake option

## Acknowledgements
- https://tfhub.dev/intel/lite-model/midas/v2_1_small/1/lite/1


