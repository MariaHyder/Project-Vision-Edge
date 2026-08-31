"""
evaluate.py
-----------
Measures how well the trained model actually performs on the full test
set (images the model never saw during training).

INPUT:  models/defect_model.keras, all images in data/test/
OUTPUT: printed accuracy, confusion matrix, and precision/recall/F1 report
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

TEST_DIR = 'data/test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)


def evaluate():
    print("Loading test data...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, batch_size=BATCH_SIZE,
        image_size=IMG_SIZE, label_mode='binary'
    )

    class_names = test_dataset.class_names
    print(f"Classes (index order): {class_names}")

    print("Loading trained model...")
    model = tf.keras.models.load_model('models/defect_model.keras')

    print("Running predictions on full test set...")
    y_true, y_scores = [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)
    y_pred = (y_scores >= 0.85).astype(int)   

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 50)
    print(f"OVERALL TEST ACCURACY: {acc:.4f}  ({int(round(acc * len(y_true)))}/{len(y_true)})")
    print("=" * 50)

    print("\nConfusion Matrix (rows=actual, cols=predicted):")
    print(f"                  Pred: {class_names[0]:<15} Pred: {class_names[1]}")
    cm = confusion_matrix(y_true, y_pred)
    for i, row in enumerate(cm):
        print(f"Actual {class_names[i]:<12} {row[0]:<22} {row[1]}")

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    margin = np.abs(y_scores - 0.85)
    borderline_idx = np.where(margin < 0.05)[0]
    print(f"Borderline predictions (score within 0.05 of the 0.85 threshold): {len(borderline_idx)} / {len(y_true)}")


if __name__ == "__main__":
    evaluate()