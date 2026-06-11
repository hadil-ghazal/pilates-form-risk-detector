

#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26


"""
Module Project CNN Training Script

Purpose:
Train a MobileNetV2 classifier on Pilates-aligned pose images
(Plank vs Downdog).

This script is separate from the original hackathon training scripts.
"""

import torch
torch.manual_seed(42) # for reporducability, forgot this in v1
from torchvision import datasets, transforms, models
from torch import nn, optim
from torch.utils.data import DataLoader, random_split

from torch.utils.data import Subset

# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/yoga_dataset/YogaPoses"
IMAGE_SIZE = 224 ##sizing images to work w MobileNetV2
BATCH_SIZE = 8 ## model to process 8 images each training batch
LEARNING_RATE = 0.001 ## starting with.001 learning rate to balance training stable training and convergence speed
NUM_EPOCHS = 5 ## will be starting w 5 complete passes through the training dataset

MODEL_SAVE_PATH = "models/pose_class_deep_learning.pth"

# -------------------------
# Image Transforms
# -------------------------

train_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)), ## resizing images for MobileNetV2 input
    transforms.RandomHorizontalFlip(), #randomly flipping images for better generalizaiotn
    transforms.RandomRotation(15), # small rotation from the hackathon project,accounting for minor camera angle differences
    transforms.ColorJitter(brightness=0.3), 
    transforms.ToTensor() 
])

test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# -------------------------
# Load Dataset
# -------------------------
# laoading entire dataset using aug for training
full_dataset = datasets.ImageFolder(
    root=DATA_DIR,
    transform=train_transforms
)
#loading same images again without augmentation
test_dataset_source = datasets.ImageFolder(
    root=DATA_DIR,
    transform=test_transforms
)

print("Classes:", full_dataset.classes)
print("Number of Images:", len(full_dataset))


# -------------------------
# Train/Test Split
# -------------------------

train_size = int(0.8 * len(full_dataset)) #using 80% of images for training will use the 20 for test
test_size = len(full_dataset) - train_size

train_subset, test_subset = random_split(
    full_dataset,
    [train_size, test_size]
)

train_dataset = train_subset

test_dataset = Subset(
    test_dataset_source,
    test_subset.indices
)

print("Training Images:", len(train_dataset))
print("Testing Images:", len(test_dataset))

# -------------------------
# Data Loaders
# -------------------------

#rrandomizing image training order
train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True 
)
#not doing the same for test, keeping image order fixed here
test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False 
)

# -------------------------
#Model Definition
# -------------------------

model = models.mobilenet_v2(pretrained=True) # Using PRETRAINED CNN on mobilenet

# Freezing the Pretrained layers to train only classifier
#aka transfer learning - using small dataset so ok
for param in model.parameters():
    param.requires_grad = False

## Replacing the original imagenet classifier with new classifier for plank/downdog
## two class classifier plank vs downdog
model.classifier[1] = nn.Linear( 
    model.last_channel,
    2
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)


# -------------------------
# Loss Function
# -------------------------


criterion = nn.CrossEntropyLoss() # using standard loss function for 2 class plank vs downdog

# -------------------------
#####  Optimizer
# -------------------------

# Choosing Adam to update model weights during training with learning rate above, flexible

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# -------------------------
# Training Loop
# -------------------------
## General setup here is: for each epoch in the 5 num epochs, the model will look at the
## .. entire training dataset once then for each batch will process 8 images and generate the predictions in line with the learning rate
for epoch in range(NUM_EPOCHS):
    running_loss = 0.0
#moving images and labels to the selected device
    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

      
        optimizer.zero_grad() # clearing old gradients before calculating new ones

        # Forward pass: To generate preds
        outputs = model(images)

        #comparing predictions against the true labels
        loss = criterion(outputs, labels) #measure how wrong the predistions were

        # Backward pass: calculating the gradients
        loss.backward() # calculating here how the weights Should change

        
        optimizer.step() # Update to the classifier weights here
        #This is where the classifier is being updated so the model is improving gradually each batch

        running_loss += loss.item() # traacking loss for the epoch

    print(f"epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {running_loss:.4f}")


# -------------------------
# Save the Trained Model
# -------------------------

#saving so it can be loaded later by the evaluation script and app
torch.save(
    model.state_dict(),
    MODEL_SAVE_PATH
)

print(f"Model saved to: {MODEL_SAVE_PATH}")

