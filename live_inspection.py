"""
live_inspection.py
-------------------
Live webcam demo. Press SPACE to inspect whatever the camera currently
sees (simulating a part arriving at the inspection point on a conveyor).
Shows a green/red border for Good/Defective, and for defective items,
overlays a heatmap showing which part of the image triggered that call
(a technique called Grad-CAM - a common way to make AI decisions
explainable, without needing a separate object-detection model).

INPUT:  models/defect_model.keras, and your webcam
OUTPUT: a live window; a simulated "gate" command printed to the terminal
        (this print() is exactly where you'd send a real command to an
        Arduino/FPGA once your conveyor hardware is built)

Controls: SPACE = inspect current frame, Q = quit
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import silence_tensorflow.auto
import tensorflow as tf
import numpy as np
import cv2

MODEL_PATH = 'models/defect_model.keras'
IMG_SIZE = (224, 224)
DEFECT_THRESHOLD = 0.85   


class DefectInspector:
    def __init__(self, model_path=MODEL_PATH):
        print("Loading model...")
        self.model = tf.keras.models.load_model(model_path)

        self.base_model = None
        self.rescale_layer = None
        for layer in self.model.layers:
            if 'mobilenet' in layer.name.lower():
                self.base_model = layer
            if 'rescaling' in layer.name.lower():
                self.rescale_layer = layer
        if self.base_model is None or self.rescale_layer is None:
            raise ValueError("Could not find expected layers in the model.")

        target_layer_name = None
        for layer in reversed(self.base_model.layers):
            try:
                if len(layer.output.shape) == 4:
                    target_layer_name = layer.name
                    break
            except Exception:
                continue

        self.feature_extractor = tf.keras.Model(
            inputs=self.base_model.input,
            outputs=self.base_model.get_layer(target_layer_name).output
        )

        base_idx = self.model.layers.index(self.base_model)
        head_layers = self.model.layers[base_idx + 1:]
        classifier_input = tf.keras.Input(shape=self.feature_extractor.output.shape[1:])
        x = classifier_input
        for layer in head_layers:
            x = layer(x)
        self.classifier_model = tf.keras.Model(classifier_input, x)

        print("Model ready.")

    def predict_with_heatmap(self, frame_bgr):
        """Takes one raw BGR frame (straight from cv2), returns (score, heatmap)."""
        img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, IMG_SIZE).astype(np.float32)
        input_data = np.expand_dims(img_resized, axis=0)

        rescaled = self.rescale_layer(input_data)

        # GradientTape records the calculations so TensorFlow can later
        # tell us how much each pixel of the feature map influenced the
        # final score - that's the core idea behind Grad-CAM.
        with tf.GradientTape() as tape:
            conv_output = self.feature_extractor(rescaled)
            tape.watch(conv_output)
            preds = self.classifier_model(conv_output)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_output_np = conv_output[0].numpy()

        heatmap = np.zeros(conv_output_np.shape[:2], dtype=np.float32)
        pooled_grads_np = pooled_grads.numpy()
        for i in range(pooled_grads_np.shape[-1]):
            heatmap += pooled_grads_np[i] * conv_output_np[:, :, i]
        heatmap = np.maximum(heatmap, 0)             # Negative values aren't useful here
        heatmap = heatmap / (np.max(heatmap) + 1e-8)  # Normalize to 0-1

        score = float(preds.numpy()[0][0])
        return score, heatmap


def overlay_heatmap(frame, heatmap, alpha=0.4):
    """Resizes the small heatmap up to the frame's size and blends it on top."""
    h, w = frame.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    return cv2.addWeighted(frame, 1 - alpha, heatmap_color, alpha, 0)


def main():
    inspector = DefectInspector()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open webcam.")
        return

    print("\nLive inspection running.")
    print("Press SPACE to inspect the current frame, Q to quit.\n")

    last_display_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: failed to grab frame.")
            break

        display = last_display_frame if last_display_frame is not None else frame.copy()
        cv2.putText(display, "SPACE = inspect | Q = quit", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imshow('Project Vision-Edge - Live Inspection', display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            score, heatmap = inspector.predict_with_heatmap(frame)
            is_defective = score < DEFECT_THRESHOLD

            result_frame = overlay_heatmap(frame.copy(), heatmap) if is_defective else frame.copy()

            border_color = (0, 0, 255) if is_defective else (0, 200, 0)   # BGR: red or green
            cv2.rectangle(result_frame, (0, 0),
                          (result_frame.shape[1] - 1, result_frame.shape[0] - 1),
                          border_color, 8)

            label = f"DEFECTIVE ({score:.3f})" if is_defective else f"GOOD ({score:.3f})"
            cv2.putText(result_frame, label, (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, border_color, 2)

            gate_status = "GATE: DIVERTING TO REJECT BIN" if is_defective else "GATE: PASS TO GOOD BIN"
            cv2.putText(result_frame, gate_status, (20, result_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, border_color, 2)
            print(f"[INSPECTION] Score={score:.4f} -> {gate_status}")

            last_display_frame = result_frame

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()