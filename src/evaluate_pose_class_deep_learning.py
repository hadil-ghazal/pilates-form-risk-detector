
#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26
"""
Evaluating the trained MobileNetV2 pose classifier using metrics like accuracy, precision, 
recall, f1 score, confusion matrix

not using augmentation during the evaluation 
"""

#pytorch libraries to load and run trained model inference on test images
import torch
torch.manual_seed(42) # for reporducability, forgot this in v1
from torchvision import datasets, transforms, models
from torch import nn
from torch.utils.data import DataLoader, random_split, Subset
#using scikit learn metrics to evaluate the model performance
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

#---------------------
# Configuration
#-------------------------

DATA_DIR = "data/yoga_dataset/YogaPoses" #full pose image dataset
MODEL_PATH = "models/pose_class_deep_learning.pth" # trained deep model here
IMAGE_SIZE = 224 #for mobilenetv2
BATCH_SIZE = 8  #8 images processing in batch


# -------------------------
# Image Transform
# -------------------------

# For evaluation we do NOT apply augmentation.
# We only resize images and convert them to tensors.
# This ensures the model is tested on the original images.
test_transforms = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor()
])

# -------------------------
# Load Dataset
# -------------------------

dataset = datasets.ImageFolder(  # loading the pose dataset using eval transforms
    root=DATA_DIR,
    transform=test_transforms
)

print("Classes:", dataset.classes)
print("Number of Images:", len(dataset))

# -------------------------
# Recreating Test Split
# -------------------------

#Recreating the same 80/20 split used during training
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

_, test_subset = random_split(
    dataset,
    [train_size, test_size]
)

# to build a DataLoader for eval here
test_loader = DataLoader(
    test_subset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print("Testing images:", len(test_subset))


# -------------------------
# Rebuild MobileNetV2 Model
# -------------------------
model = models.mobilenet_v2(pretrained=False) # mirroring the same MNv2 archiitecture used in train step

## Replacing the original imagenet classifier with new classifier for plank/downdog
## two class classifier plank vs downdog
model.classifier[1] = nn.Linear(
    model.last_channel, 
    2
)

#Selecting GPU or cpu
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

#Loading the trained model weights from disk
model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=device
    )
)

model = model.to(device) #moving to device

model.eval() #putting the model into eval mode to disable training behavior

print("Model loaded success")




# -------------------------
# Generate Predictoins 
# -------------------------


all_labels = [] #lists for storing labels
all_predictions = [] #and predicitons

# Disabled gradient cal because only evaluating the model itself
with torch.no_grad():

    for images, labels in test_loader:

       
        images = images.to(device) #moving images to device
        outputs = model(images) #generating model preditions

    
        _, predictions = torch.max(outputs, 1) # choosing the class that has highest score

        #storing the labels and preds for metrics
        all_labels.extend(labels.numpy())
        all_predictions.extend(predictions.cpu().numpy())

# -------------------------
# Evaluation Metrics
# -------------------------

# Overall percentage of correct/good predics
accuracy = accuracy_score(
    all_labels,
    all_predictions
)


precision = precision_score( # measuring how many # positive preds that were actually right
    all_labels,
    all_predictions,
    average="weighted"
)


recall = recall_score( #measuring how many # true examples were correctly identified
    all_labels,
    all_predictions,
    average="weighted"
)

#F1 score to balance precision and recall
f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted"
)


cm = confusion_matrix( # conf matrix showing correct vs incorrect preds by class combo
    all_labels,
    all_predictions
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)
