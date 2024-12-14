# HRNet Swimming Pose Estimation

## Overview
This project focuses on adapting a High-Resolution Network (HRNet) for pose estimation in an underwater swimming environment. Pose estimation underwater presents unique challenges, including light refraction, motion distortion, and occlusions caused by water turbulence. 
Leveraging HRNet's ability to maintain high-resolution representations throughout the process, this project aims to overcome these obstacles and achieve accurate keypoint detection.

## Data Collection
The dataset for this project was collected from the **University of South Carolina Swim and Dive Team**. The athletes' performances were recorded during training sessions to generate a comprehensive dataset for underwater pose estimation. 
You can learn more about the swim and dive team [here](https://gamecocksonline.com/sports/swimming/).
<br>The dataset used for training can be found in the data directory. it contains a few dataset directories, each from a different video. each dataset will have a directory for the frames and the COCO format JSON annotation file.
each frame is annotated with 13 keypoints to represent swimmer biomechanics.


## Implementation Details

This project was developed based on the research paper **"[Deep High-Resolution Representation Learning for Human Pose Estimation](https://openaccess.thecvf.com/content_CVPR_2019/papers/Sun_Deep_High-Resolution_Representation_Learning_for_Human_Pose_Estimation_CVPR_2019_paper.pdf)"** 
by Ke Sun, Bin Xiao, Dong Liu, and Jingdong Wang. The paper introduces HRNet, which maintains high-resolution representations throughout the process, achieving superior accuracy in pose estimation tasks.
This project's implementation draws inspiration from Bin Xiao's official HRNet repository, which is available on GitHub [here](https://github.com/leoxiaobin/deep-high-resolution-net.pytorch?tab=readme-ov-file).

## Installation

1. Clone this repository to your local machine:
    ```bash
    git clone https://github.com/markshperkin/SwimmingPoseEstimation.git
    ```

2. Navigate into the project directory:
    ```bash
    cd SwimmingPoseEstimation
    ```

3. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Testing the Model on a Swimming Video

First, open `Tester.py` and add the model and input video file paths in lines 182 and 188. <br>Next, run the `Tester.py` script. 
```bash
python train_model.py
```
This script will estimate keypoints found in the video and output a new video with the estimated keypoints.

### 2. Training the Model

Training the model will require NVidia GPU. In order to modify the learning rate, batch size, and the number of epochs go to lines 142 and 145. After a new minimum has been found, the trainer will save the model as a checkpoint.pth.
<br> To initialize the training, run the `hrnetTrainer.py`
```bash
python hrnetTrainer.py
```
## 3. System Testers
There are a few scripts to test various components of the system.
### Test the Model on One Frame:
The `hrnetTester.pt` script will test the accuracy of the model by loading a single frame on a desired model and showing it with the corresponding keypoints, the target ground truth for each keypoint (GT keypoint n), the confidence level for each predicted keypoint, 
the ground truth for each predicted keypoint (Pred Keypoint n), and the Euclidean distance from the target keypoint to the predicted keypoint.
<br>First, open `hrnetTester.pt` and add the desired model and frame for testing on lines 165 and 174. Next run `hrnetTester.py.
```bash
python hrnetTester.py
```
### Test the Data Augmentation Functionality:
The `testerAugmentation.py` script will test three data augmentation techniques, Horizontal Flip, Rotation, and Translation. The function will select a random frame from the dataset and apply the augmentations, showing the before and after frames with the corresponding keypoints.
<br> To run the augmentation tester:
```bash
python testerAugmentation.py


```
### Test the Data Loader:
The `testerDATALOADER.py` script will test the functionality of the dataloader by loading the dataset and displaying a batch of 4 frames with their name and corresponding keypoints.
<br> To run the dataloader tester:
```bash
python testerDATALOADER.py
```


## Class Project

This project was developed as part of the Neural Network class under the instruction of [Professor Vignesh Narayanan](https://sc.edu/study/colleges_schools/engineering_and_computing/faculty-staff/narayanan_vignesh.php) at the University of South Carolina.
