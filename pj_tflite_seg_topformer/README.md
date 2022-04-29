# Semantic Segmentation using TopFormer with TensorFlow Lite in C++


## How to Run
1. Please follow the instruction: https://github.com/iwatake2222/play_with_tflite/blob/master/README.md
2. Additional steps:
    - Download the model using the following script
        - https://github.com/PINTO0309/PINTO_model_zoo/blob/main/287_Topformer/download.sh
        - copy `topformer_tiny_448x448/model_float32.tflite` to `resource/model/topformer_tiny_448x448.tflite`
    - Build  `pj_tflite_seg_topformer` project (this directory)

### Tested environment
- Windows 11
    - Core i7-11700 @ 2.5GHz x 8 cores (16 processors)
    - Visual Studio 2019
- Raspberry Pi 4
    - 8.8 FPS (less than 100msec inference time) with Camera input

## Acknowledgements
- https://github.com/hustvl/TopFormer
- https://github.com/PINTO0309/PINTO_model_zoo
- Test image
    - Drive Video by Dashcam Roadshow
    - 4K 首都高ドライブ横浜みなとみらい→ベイブリッジ→横羽線→渋谷44km
    - https://www.youtube.com/watch?v=TGeoG5l3f38
