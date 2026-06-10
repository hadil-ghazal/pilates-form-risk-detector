
#No AI was used to generate this code, authored by Hadil Ghazal 6/9/26
"""
Random Forest Classical ML Model: will be traiing a classical machine learning model for comparison
against the Naive Baseline and Deep Learning model
steps include: loading pose images, resizing them, flattening pixel values into the feature vectors,
training random forest classifier, saving results for comparison

"""

import os
import pickle
import numpy as np

from PIL import Image

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -------------------------
# Configuration
# -------------------------

DATA_DIR = "data/yoga_dataset/YogaPoses"

IMAGE_SIZE = (64, 64)

MODEL_SAVE_PATH = "models/random_forest_classical_ml.pkl"

RANDOM_STATE = 42


#-----------------------
#### Load Images
# -------------------------


X = [] #feature matrix here


y = [] #labels

# Class folders
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

            
            image_features = image_array.flatten() #flattening the image into a 1D vector

            X.append(image_features)

            y.append(label_index)

        except Exception:

            continue



X = np.array(X) #Conversion to numpy arrays
y = np.array(y)

print("Images Loaded:", len(X))


# -------------------------
# Train And Test split
#---------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y
)


# -------------------------
# Train Random Forest
# -------------------------

rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=RANDOM_STATE
)

rf_model.fit(
    X_train,
    y_train
)



#Model Evaliation
#-------------------------

predictions = rf_model.predict(
    X_test
)

accuracy = accuracy_score(
    y_test,
    predictions
)

print(f"Accuracy: {accuracy:.4f}")


# -------------------------
# Saving Model
# -------------------------

with open(
    MODEL_SAVE_PATH,
    "wb"
) as file:

    pickle.dump(
        rf_model,
        file
    )

print(
    f"Model saved to: {MODEL_SAVE_PATH}" #tracking where saved to
)
