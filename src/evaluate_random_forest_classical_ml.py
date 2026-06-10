#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26

"""
Evaluating Random Forest Classical ML Model
Section to assess the trained Random Forest model using:
Accuracy, Precision, Recall, f1 score and Confusion Matrix
"""

import os
import pickle
import numpy as np

from PIL import Image

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from sklearn.model_selection import train_test_split

# ----------------
# Configuration Step
# -----------------

DATA_DIR = "data/yoga_dataset/YogaPoses"

IMAGE_SIZE = (64, 64)

MODEL_PATH = "models/random_forest_classical_ml.pkl"

RANDOM_STATE = 42

# -----------------
# Loading Images
# ------------------

X = []
y = []

classes = ["Downdog", "Plank"]

for label_index, class_name in enumerate(classes):

    class_folder = os.path.join(
        DATA_DIR,
        class_name
    )

    for filename in os.listdir(class_folder):

        image_path = os.path.join(
            class_folder,
            filename
        )

        try:

            image = Image.open(image_path)

            image = image.convert("RGB")

            image = image.resize(
                IMAGE_SIZE
            )

            image_array = np.array(
                image
            )

            X.append(
                image_array.flatten()
            )

            y.append(
                label_index
            )

        except Exception:
            continue

X = np.array(X)
y = np.array(y)

# ————————————s
## using the same split used in training
# -------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)

# -------------------------
# Loading Model
# -------------------------

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)

# -------------------------
# Predictions
# -------------------------

predictions = model.predict(X_test)

# -------------------------
# Metrics
# -------------------------

accuracy = accuracy_score(y_test, predictions)

precision = precision_score(
    y_test,
    predictions,
    average="weighted"
)

recall = recall_score(
    y_test,
    predictions,
    average="weighted"
)

f1 = f1_score(
    y_test,
    predictions,
    average="weighted"
)

cm = confusion_matrix(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)