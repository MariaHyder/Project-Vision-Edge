"""
convert_tflite.py
------------------
Converts the trained Keras model into a smaller TensorFlow Lite file for
fast testing on your PC/laptop. Run this AFTER train.py (or fine_tune.py).

INPUT:  models/defect_model.keras
OUTPUT: models/optimized_defect.tflite

NOTE: This uses "dynamic-range" quantization - only the model's WEIGHTS
are compressed to int8; the actual math still runs in float32. This is
fine for testing on a PC, but most microcontrollers/FPGAs need FULL int8
(weights AND math). For that, use convert_tflite_int8.py instead.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf

MODEL_PATH = 'models/defect_model.keras'
OUTPUT_PATH = 'models/optimized_defect.tflite'


def convert_to_tflite():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train.py first.")
        return

    print(f"Loading trained model from '{MODEL_PATH}'...")
    # Loading the FULL saved model (architecture + weights together) is
    # simpler and safer than rebuilding the architecture and loading
    # weights separately - there's no risk of silently mismatched layers.
    model = tf.keras.models.load_model(MODEL_PATH)
    print("Model loaded successfully!")

    print("Converting model to TensorFlow Lite (dynamic-range quantized)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    size_mb = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSuccess! Saved to: {OUTPUT_PATH}  ({size_mb:.2f} MB)")


if __name__ == "__main__":
    convert_to_tflite()