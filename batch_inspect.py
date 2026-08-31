"""
batch_inspect.py
-----------------
Runs the model over an ENTIRE FOLDER of images at once (simulating a
batch of parts coming off the line) and produces a CSV inspection report
plus a printed summary. This is useful for your demo day (show a folder
of test images being inspected all at once) and gives you a concrete
"performance report".

INPUT:  models/optimized_defect.tflite, and a folder of images
        (default: data/batch_inspect - create this folder and put any
        mix of images in it)
OUTPUT: models... no - outputs/inspection_report_<timestamp>.csv, plus
        a printed summary in the terminal

Usage:
    python src/batch_inspect.py
    python src/batch_inspect.py path/to/your/folder
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import sys
import csv
import time
import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from preprocessing import preprocess_inspection_frame

MODEL_PATH = 'models/optimized_defect.tflite'
DEFECT_THRESHOLD = 0.85    
BORDERLINE_MARGIN = 0.1    
VALID_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp')


def run_batch_inspection(folder_path):
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run convert_tflite.py first.")
        return

    if not os.path.isdir(folder_path):
        print(f"Error: folder not found: '{folder_path}'")
        print("Create this folder and add some images, or pass a different path:")
        print("  python src/batch_inspect.py path/to/your/folder")
        return

    image_files = sorted(
        f for f in os.listdir(folder_path)
        if f.lower().endswith(VALID_EXTENSIONS)
    )
    if not image_files:
        print(f"No images found in '{folder_path}'.")
        return

    print(f"Found {len(image_files)} images in '{folder_path}'. Loading model...")
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    results = []
    print("Inspecting...")
    for filename in image_files:
        image_path = os.path.join(folder_path, filename)
        try:
            processed_img = preprocess_inspection_frame(image_path, target_size=(224, 224))
        except FileNotFoundError:
            continue

        input_data = np.expand_dims(processed_img, axis=0).astype(np.float32)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        score = float(interpreter.get_tensor(output_details[0]['index'])[0][0])

        verdict = "DEFECTIVE" if score < DEFECT_THRESHOLD else "GOOD"
        is_borderline = abs(score - DEFECT_THRESHOLD) < BORDERLINE_MARGIN
        flag = "REVIEW RECOMMENDED" if is_borderline else ""

        results.append({
            "filename": filename,
            "score": round(score, 4),
            "verdict": verdict,
            "flag": flag
        })

    # --- Write the CSV report ---
    os.makedirs('reports', exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    report_path = os.path.join('reports', f'inspection_report_{timestamp}.csv')
    with open(report_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "score", "verdict", "flag"])
        writer.writeheader()
        writer.writerows(results)

    # --- Print the summary ---
    total = len(results)
    good_count = sum(1 for r in results if r["verdict"] == "GOOD")
    defective_count = total - good_count
    borderline_count = sum(1 for r in results if r["flag"])

    print("\n" + "=" * 50)
    print("BATCH INSPECTION SUMMARY")
    print("=" * 50)
    print(f"Total items inspected:  {total}")
    print(f"Good:                   {good_count}  ({100 * good_count / total:.1f}%)")
    print(f"Defective:              {defective_count}  ({100 * defective_count / total:.1f}%)")
    print(f"Flagged for review:     {borderline_count}")
    print(f"\nFull report saved to: {report_path}")


if __name__ == "__main__":
    folder = sys.argv[1] if len(sys.argv) > 1 else 'data/batch_inspect'
    run_batch_inspection(folder)