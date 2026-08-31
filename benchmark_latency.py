"""
benchmark_latency.py
---------------------
Measures how many milliseconds each model takes to process one image, so
you can check against your proposal's real-time speed target.

INPUT:  models/optimized_defect.tflite and/or models/optimized_defect_int8.tflite
OUTPUT: printed average and 95th-percentile latency for each model found
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import time

NUM_RUNS = 50
TARGET_MS = 50   


def benchmark(model_path, is_int8=False):
    if not os.path.exists(model_path):
        print(f"  Skipping - '{model_path}' not found.")
        return

    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # We use random dummy data here on purpose - we're timing the MODEL's
    # speed, not testing its accuracy (that's what evaluate.py is for).
    if is_int8:
        dummy_input = np.random.randint(-128, 127, size=(1, 224, 224, 3), dtype=np.int8)
    else:
        dummy_input = np.random.uniform(0, 255, size=(1, 224, 224, 3)).astype(np.float32)

    interpreter.set_tensor(input_details[0]['index'], dummy_input)
    interpreter.invoke()

    times_ms = []
    for _ in range(NUM_RUNS):
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]['index'], dummy_input)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]['index'])
        times_ms.append((time.perf_counter() - start) * 1000)

    avg_ms = np.mean(times_ms)
    p95_ms = np.percentile(times_ms, 95)
    status = "PASS" if avg_ms < TARGET_MS else "FAIL"
    print(f"  Average: {avg_ms:.2f} ms  |  95th percentile: {p95_ms:.2f} ms  |  Target <{TARGET_MS}ms: {status}")


def main():
    print(f"Benchmarking inference latency ({NUM_RUNS} runs each, on this machine's CPU)\n")
    print("NOTE: This is a CPU baseline on your PC, not the real MCU/FPGA number -")
    print("but it's a legitimate data point for your report and a useful sanity")
    print("check before committing to hardware.\n")

    print("Dynamic-range quantized model (optimized_defect.tflite):")
    benchmark('models/optimized_defect.tflite', is_int8=False)

    print("\nFull INT8 quantized model (optimized_defect_int8.tflite):")
    benchmark('models/optimized_defect_int8.tflite', is_int8=True)


if __name__ == "__main__":
    main()