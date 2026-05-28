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
