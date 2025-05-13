import numpy as np
import os
import cv2
from keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from pathlib import Path

#match class names with those used in training
IMG_SIZE = 227
TEST_DIR = Path("image_data")  # same structure as training dir
CLASS_NAMES = sorted([d.name for d in TEST_DIR.iterdir() if d.is_dir()])
OUTPUT_FILE = "metrics_report.txt"

#load test data
def load_images(image_dir):
    images = []
    labels = []
    for label in CLASS_NAMES:
        class_path = image_dir / label
        for img_file in tqdm(list(class_path.glob("*.jpg")), desc=f"Loading {label}"):
            try:
                img = cv2.imread(str(img_file))
                img = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (IMG_SIZE, IMG_SIZE))
                images.append(img)
                labels.append(CLASS_NAMES.index(label))
            except:
                print(f"⚠️ Skipped unreadable file: {img_file}")
    return np.array(images), np.array(labels)

# Load model and data
print("🔍 Loading model and test data...")
model = load_model("rock-paper-scissors-model.h5")
X_test, y_true = load_images(TEST_DIR)
X_test = X_test / 255.0  # Normalize

#predict
y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

#calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
conf_matrix = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)

#save to file
with open(OUTPUT_FILE, "w") as f:
    f.write("Rock-Paper-Scissors Model Evaluation Report\n")
    f.write("=" * 50 + "\n\n")
    f.write(f"Accuracy: {accuracy:.4f}\n")
    f.write(f"Precision (macro avg): {precision:.4f}\n")
    f.write(f"Recall (macro avg): {recall:.4f}\n")
    f.write(f"F1 Score (macro avg): {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(conf_matrix) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

print(f"\nEvaluation complete. Metrics saved to {OUTPUT_FILE}")
