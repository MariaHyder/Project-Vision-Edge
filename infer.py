"""
infer.py
--------
Tests the (dynamic-range quantized) TFLite model on a single image file
and prints Good/Defective. Good for quickly checking one image at a time.

INPUT:  models/optimized_defect.tflite, and one image path
OUTPUT: printed score and Good/Defective verdict
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
from preprocessing import preprocess_inspection_frame

DEFECT_THRESHOLD = 0.85


def predict_casting_defect(image_path, model_path='models/optimized_defect.tflite',
                            threshold=DEFECT_THRESHOLD):
    if not os.path.exists(model_path):
        print(f"Error: TFLite model not found at '{model_path}'. Run convert_tflite.py first!")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at path: {image_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    processed_img = preprocess_inspection_frame(image_path, target_size=(224, 224))
    input_data = np.expand_dims(processed_img, axis=0).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])
    confidence_score = float(output_data[0][0])

    print("\n" + "=" * 40)
    print("   PROJECT VISION-EDGE: INSPECTION   ")
    print("=" * 40)
    print(f"Target Image: {image_path}")
    print(f"Model Raw Score: {confidence_score:.4f}   (threshold: {threshold})")

    if confidence_score < threshold:
        print("Status: DEFECTIVE PART DETECTED")
    else:
        print("Status: GOOD PART (Flawless)")
    print("=" * 40 + "\n")

    return confidence_score


if __name__ == "__main__":
    sample_image = "data/test/good_parts/cast_ok_0_505.jpeg"   # <-- change this to test other images
    print("Inference module initialized.")
    predict_casting_defect(sample_image)