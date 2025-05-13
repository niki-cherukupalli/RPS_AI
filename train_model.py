import cv2
import numpy as np
from pathlib import Path
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dropout, Conv2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.optimizers import Adam

# path for data images
IMG_SAVE_PATH = Path('image_data')

CLASS_NAMES = sorted([d.name for d in IMG_SAVE_PATH.iterdir() if d.is_dir()])
NUM_CLASSES = len(CLASS_NAMES)

def get_model():
    base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
    base_model.trainable = False  # Freeze the base model
    return Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation='softmax')
    ])

# load and go through images
dataset = []
for label_dir in IMG_SAVE_PATH.iterdir():
    
    if not label_dir.is_dir():
        continue
    
    for img_path in label_dir.iterdir():
        if img_path.name.startswith('.'):
            continue
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        dataset.append((img, label_dir.name))

data, labels = zip(*dataset)
label_indices = [CLASS_NAMES.index(label) for label in labels]
labels_categorical = to_categorical(label_indices, NUM_CLASSES)

#training the model
model = get_model()
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(np.array(data), labels_categorical, epochs=10)


model.save("rock-paper-scissors-model.h5")
print("Model saved as rock-paper-scissors-model.h5")
