"""
convert_tflite_int8.py
-----------------------
Converts the trained model into a FULLY int8-quantized TensorFlow Lite
file - both weights AND the internal math run in int8. This is what most
microcontroller (TFLite Micro) and FPGA (Vitis-AI) toolchains actually
require for embedded deployment.

INPUT:  models/defect_model.keras
OUTPUT: models/optimized_defect_int8.tflite

Because this model expects INT8 input (not float32), you must use
infer_int8.py (not infer.py) to test it.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf

TRAIN_DIR = 'data/train'
IMG_SIZE = (224, 224)
MODEL_PATH = 'models/defect_model.keras'
OUTPUT_PATH = 'models/optimized_defect_int8.tflite'
NUM_CALIBRATION_SAMPLES = 100


def representative_dataset():
    """
    Full int8 quantization needs a small sample of REAL images to measure
    the typical range of values flowing through the network, so it can
    pick good int8 scaling factors for every layer. This is called
    'calibration'. We reuse training images for this - no labels needed.
    """
    calib_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,
        batch_size=1,
        image_size=IMG_SIZE,
        label_mode=None   
    )
    for image_batch in calib_dataset.take(NUM_CALIBRATION_SAMPLES):
        
        yield [tf.cast(image_batch, tf.float32)]


def convert_full_int8():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: '{MODEL_PATH}' not found. Run train.py or fine_tune.py first.")
        return

    print(f"Loading trained model from '{MODEL_PATH}'...")
    model = tf.keras.models.load_model(MODEL_PATH)

    print(f"Calibrating on {NUM_CALIBRATION_SAMPLES} sample images and converting to full INT8...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
   
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, 'wb') as f:
        f.write(tflite_model)

    original_size = os.path.getsize(MODEL_PATH) / (1024 * 1024)
    int8_size = os.path.getsize(OUTPUT_PATH) / (1024 * 1024)
    print(f"\nSuccess! Full INT8 model saved to: {OUTPUT_PATH}")
      print(f"Original .keras size: {original_size:.2f} MB")
    print(f"INT8 .tflite size:  {int8_size:.2f} MB  ({original_size/int8_size:.1f}x smaller)")
    print("\nRemember: this model needs INT8 input. Use infer_int8.py to test it,")
    print("not infer.py (which is for the other, float32-input .tflite file).")


if __name__ == "__main__":
    convert_full_int8()