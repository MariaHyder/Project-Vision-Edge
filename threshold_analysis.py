"""
threshold_analysis.py
----------------------
The model outputs a score from 0 to 1, and we need to pick a cutoff point
("threshold") to decide Good vs Defective. 0.5 is the default, but it's
not always the best choice - this script tests many thresholds and shows
you which one minimizes missed defects (the costly error for a quality
control system) without creating too many false alarms.

INPUT:  models/defect_model.keras, all images in data/test/
OUTPUT: a printed table comparing accuracy/missed-defects/false-alarms
        at each threshold value

Run this AFTER evaluate.py. Once you pick a threshold from the table,
update DEFECT_THRESHOLD in infer.py, infer_int8.py, and live_inspection.py
to match.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score

TEST_DIR = 'data/test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
THRESHOLDS_TO_TEST = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60,
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def evaluate_at_threshold(y_true, y_scores, threshold):
    y_pred = (y_scores >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    # class 0 = defective_parts, class 1 = good_parts
    missed_defects = cm[0][1]  
    false_alarms = cm[1][0]     
    defect_recall = cm[0][0] / (cm[0][0] + cm[0][1]) if (cm[0][0] + cm[0][1]) > 0 else 0
    return acc, missed_defects, false_alarms, defect_recall


def main():
    print("Loading test data...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, batch_size=BATCH_SIZE,
        image_size=IMG_SIZE, label_mode='binary'
    )

    print("Loading model...")
    model = tf.keras.models.load_model('models/defect_model.keras')

    print("Running predictions on full test set...\n")
    y_true, y_scores = [], []
    for images, labels in test_dataset:
        preds = model.predict(images, verbose=0)
        y_scores.extend(preds.flatten().tolist())
        y_true.extend(labels.numpy().flatten().tolist())

    y_true = np.array(y_true)
    y_scores = np.array(y_scores)

    print(f"{'Threshold':<12}{'Accuracy':<12}{'Missed Defects':<18}{'False Alarms':<15}{'Defect Recall'}")
    print("-" * 75)
    best_threshold, best_missed = 0.5, float('inf')
    for t in THRESHOLDS_TO_TEST:
        acc, missed, false_alarms, recall = evaluate_at_threshold(y_true, y_scores, t)
        marker = "  <-- default" if t == 0.50 else ""
        print(f"{t:<12}{acc:<12.4f}{missed:<18}{false_alarms:<15}{recall:.4f}{marker}")
        if missed < best_missed:
            best_missed, best_threshold = missed, t

    print(f"\nThreshold with FEWEST missed defects: {best_threshold} ({best_missed} missed)")
    print("\nHow to read this: raising the threshold means the model needs a HIGHER")
    print("score to call something 'good', so more borderline cases get flagged as")
    print("defective instead. This usually trades a few more false alarms for fewer")
    print("missed defects, which is normally the right tradeoff for quality control.")
    print("Pick the threshold that best balances your accuracy target against how")
    print("many false alarms you're willing to accept, then update DEFECT_THRESHOLD")
    print("in infer.py, infer_int8.py, and live_inspection.py to match.")


if __name__ == "__main__":
    main()