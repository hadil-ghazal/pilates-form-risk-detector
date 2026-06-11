# Project Update: Module Project Enhancements

## Overview

This project evolved from the original Pilates Form Risk Detector hackathon prototype into a broader computer vision posture classification system capable of identifying yoga and Pilates inspired movement patterns using transfer learning and classical machine learning approaches

## New Enhancements

### Computer Vision Model Development

* Implemented a majority class naive baseline for benchmarking 
* Developed a Random Forest image classification pipeline 
* Developed a MobileNetV2 transfer learning model for pose classification 
* Evaluated models using accuracy, precision, recall, F1-score, and confusion matrix analysis 
* Compared model performance across baseline, classical machine learning, and deep learning approaches

### Pose Classification Expansion

The project was expanded beyond safe vs. risky posture detection to include pose recognition:

* Downdog
* Plank

This allows the application to identify pose type before performing additional posture assessment

### Application Enhancements:

* Integrated the final MobileNetV2 model into the Gradio application
* Added prediction confidence scoring.
* Added pose specific feedback and recommendations
* Added an additional risk assessment layer for plank posture evaluation 
* Improved output formatting and user experience
## Key Findings:

| Model                         | Accuracy |
| ----------------------------- | -------- |
| Naive Baseline                | 50.13%   |
| Random Forest                 | 86.08%   |
| MobileNetV2 Transfer Learning | 92.41%   |

MobileNetV2 achieved the highest overall performance and was selected as the final deployment model

## Future Work

While this project honed in on pose detection, layering in risk analysis was tricky, some enhancements to focus on:
* Accurate risk detection with planks (correcting overfitting issue)
* Additional Pilates and yoga pose categories
* Realtime webcam posture analysis
* Expanded risk assessment across multiple exercises.
* Larger and more diverse training datasets
* Personalized movement feedback and coaching recommendations


## RISK DETECTION OVERFITTING ISSUE: Limitations and Observations

An additional risk assessment layer was explored by reusing the original hackathon Safe Form vs. Risky Form classifier. While the model performed well on the small original prototype dataset, it did not generalize effectively to the expanded pose classification dataset used in the module project. Testing revealed that the risk model frequently produced inconsistent predictions when applied to new plank images outside of its original training distribution. This behavior suggests overfitting to the small set of manually collected training examples, causing the model to learn dataset-specific visual characteristics rather than generalizable indicators of posture quality. As a result, the MobileNetV2 pose classification model demonstrated strong performance for pose recognition, while the risk assessment component would require a larger and more diverse labeled dataset before being considered reliable for real-world posture safety evaluation.


## How to Run

### Option 1: Run the Deployed Application

Access the deployed Gradio application through Hugging Face Space: https://huggingface.co/spaces/hadilghazal/pilates-form-risk-detector

### Option 2: Run Locally

#### 1. Install Dependencies on bash
python -m pip install -r requirements.txt


#### 2. Launch the Application


python app/app.py


#### 3. Open the Local URL

After launching, Gradio will generate a local URL similar to: http://127.0.0.1:7860


Open the URL in a web browser and upload an image for inference




### Reproducing Model Training

#### Train MobileNetV2 Pose Classification Model
python src/train_pose_class_deep_learning.py


#### Train Random Forest Baseline

python src/train_random_forest_classical_ml.py


#### Run Model Evaluation


python src/evaluate_pose_class_deep_learning.py


#### Compare Model Performance


python src/compare_model_results.py


### Application Features

The final application supports:

* Pose classification (Downdog verss Plank)
* Prediction confidence display
* Risk assessment layer for plank posture analysis
* Gradio based user interface for image upload and inference





---
---
---

# PREVIOUS VERSION: Archived as of 6/10/26
---
title: Pilates Form Risk Detector
sdk: gradio
sdk_version: "4.44.1"
python_version: "3.8"
app_file: app/app.py
pinned: false
---

# pilates-form-risk-detector
Exploring computer vision for real-time movement analysis, injury risk detection, and human performance monitoring through a Pilates posture classification prototype using transfer learning and data augmentation.

# Pilates Form Risk Detector
Mini Hackathon #1: How Can Machines See What Matters?

## Problem Statement
In group Pilates classes, instructors cannot continuously monitor every student at once. This prototype explores whether transfer learning and data augmentation can help classify Pilates posture images as either safe form or potentially risky form. The end product will augment class monitoring, where instead of 100% of students being watched 30% of the time, all students will be monitored 100% of the time. 

## Classes
- Safe Form
- Potentially Risky Form

## Model
This project uses MobileNetV2 as a pre-trained CNN model. The base model was frozen, and only the final classification layer was retrained for the two Pilates form classes.

## Data
A small prototype dataset was collected manually with images of Pilates poses, including plank and hundred examples. The dataset is intentionally small to reflect limited real-world data.

## My Approach to Augmenting
Augmentations were chosen to simulate realistic Pilates studio variation:

- horizontal flipping for mirrored studio positioning
- random rotation for camera angle variation
- brightness changes for lighting variation
- resizing/cropping for camera framing variation

## Experiment

Two training scripts were created:

- `src/train_no_aug.py`: trains without augmentation
- `src/train_with_aug.py`: trains with augmentation

This allows comparison between a baseline model and an augmented model.

## Demo
HuggingFace: https://huggingface.co/spaces/hadilghazal/pilates-form-risk-detector
Gradio app: https://830ded73747abf5cb4.gradio.live

## How to Run

### Option 1 )Recreate Full Project
- Install dependencies in bash terminal using: python -m pip install -r requirements.txt
- add training images into /data folder in corresponding "risky" and "good" form folders
- train baseline model using: python src/train_no_aug.py
- train enhanced model using: python src/train_with_aug.py
- run app using: python app/app.py
### Option 2) Run Existing model only
- run app using: python app/app.py
