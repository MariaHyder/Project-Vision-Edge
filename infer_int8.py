"""
infer_int8.py
-------------
Same job as infer.py, but for the FULL INT8 quantized model
(models/optimized_defect_int8.tflite). This model needs its input
converted into int8 numbers and its output converted back - that
quantize/dequantize math is the main difference from infer.py.

INPUT:  models/optimized_defect_int8.tflite, and one image path
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


def predict_casting_defect_int8(image_path, model_path='models/optimized_defect_int8.tflite',
                                 threshold=DEFECT_THRESHOLD):
    if not os.path.exists(model_path):
        print(f"Error: INT8 TFLite model not found at '{model_path}'. Run convert_tflite_int8.py first!")
        return

    if not os.path.exists(image_path):
        print(f"Error: Image not found at path: {image_path}")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']

    processed_img = preprocess_inspection_frame(image_path, target_size=(224, 224))
    input_data = np.expand_dims(processed_img, axis=0)

    input_data_int8 = (input_data / input_scale + input_zero_point).astype(np.int8)

    interpreter.set_tensor(input_details[0]['index'], input_data_int8)
    interpreter.invoke()

    output_data_int8 = interpreter.get_tensor(output_details[0]['index'])

    confidence_score = float((output_data_int8[0][0].astype(np.float32) - output_zero_point) * output_scale)

    print("\n" + "=" * 40)
    print("  PROJECT VISION-EDGE: INSPECTION (INT8)  ")
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
    print("INT8 inference module initialized.")
    predict_casting_defect_int8(sample_image)