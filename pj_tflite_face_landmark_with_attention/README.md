# FaceMesh with TensorFlow Lite in C++
Sample project to run FaceMesh

![00_doc/facemesh.jpg](00_doc/facemesh.jpg)

## Target Environment, How to Build, How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the face detection model
        - https://github.com/iwatake2222/play_with_tflite/blob/master/pj_tflite_face_blazeface
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/032_FaceMesh/01_float32/download.sh
        - copy `face_landmark.tflite` to `resource/model/face_landmark.tflite`
    - Build  `pj_tflite_face_facemesh` project (this directory)

## Acknowledgements
- https://github.com/google/mediapipe
- https://github.com/PINTO0309/PINTO_model_zoo
