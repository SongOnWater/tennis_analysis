# Tennis Analysis

## Introduction
This project analyzes Tennis players in a video to measure their speed, ball shot speed and number of shots. This project will detect players and the tennis ball using YOLO and also utilizes CNNs to extract court keypoints. This hands on project is perfect for polishing your machine learning, and computer vision skills. 

## Output Videos
Here is a screenshot from one of the output videos:

![Screenshot](output_videos/screenshot.jpeg)

## Models Used
* YOLO v8 for player detection
* Fine Tuned YOLO for tennis ball detection
* Court Key point extraction

* Trained YOLOV5 model: https://drive.google.com/file/d/1UZwiG1jkWgce9lNhxJ2L0NVjX1vGM05U/view?usp=sharing
* Trained tennis court key point model: https://drive.google.com/file/d/1QrTOF1ToQ4plsSZbkBs3zOLkVt3MBlta/view?usp=sharing

## Training
* Tennis ball detetcor with YOLO: training/tennis_ball_detector_training.ipynb
* Tennis court keypoint with Pytorch: training/tennis_court_keypoints_training.ipynb

## Requirements
* python==3.8
* ultralytics==8.3.176
* torch==2.8.0+cu126
* pandas==2.3.1
* numpy==2.1.2
* opencv-python==4.12.0.88

### Full Environment Package List
Package                Version
---------------------- ------------
adam_atan2             0.0.3
annotated-types        0.7.0
antlr4-python3-runtime 4.9.3
argdantic              1.3.3
certifi                2025.8.3
charset-normalizer     3.4.3
click                  8.2.1
colorama               0.4.6
contourpy              1.3.3
coolname               2.2.0
cycler                 0.12.1
einops                 0.8.1
filelock               3.13.1
fonttools              4.59.0
fsspec                 2024.6.1
gitdb                  4.0.12
GitPython              3.1.45
huggingface-hub        0.34.4
hydra-core             1.3.2
idna                   3.10
Jinja2                 3.1.4
kiwisolver             1.4.9
lap                    0.5.12
MarkupSafe             2.1.5
matplotlib             3.10.5
mpmath                 1.3.0
networkx               3.3
numpy                  2.1.2
omegaconf              2.3.0
opencv-python          4.12.0.88
packaging              25.0
pandas                 2.3.1
pillow                 11.0.0
pip                    25.1.1
platformdirs           4.3.8
protobuf               6.31.1
psutil                 7.0.0
py-cpuinfo             9.0.0
pydantic               2.11.7
pydantic_core          2.33.2
pydantic-settings      2.10.1
pyparsing              3.2.3
python-dateutil        2.9.0.post0
python-dotenv          1.1.1
pytz                   2025.2
PyYAML                 6.0.2
requests               2.32.4
scipy                  1.16.1
sentry-sdk             2.34.1
setuptools             70.2.0
six                    1.17.0
smmap                  5.0.2
sympy                  1.13.3
torch                  2.8.0+cu126
torchaudio             2.8.0+cu126
torchvision            0.23.0+cu126
tqdm                   4.67.1
typing_extensions      4.12.2
typing-inspection      0.4.1
tzdata                 2025.2
ultralytics            8.3.176
ultralytics-thop       2.0.15
urllib3                2.5.0
wandb                  0.21.1