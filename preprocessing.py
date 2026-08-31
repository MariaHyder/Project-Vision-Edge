

import cv2                     # OpenCV - reads and resizes images
import numpy as np             # Handles the image as a numerical array


def preprocess_inspection_frame(image_path, target_size=(224, 224)):
    """
    Reads an image from disk and prepares it for the model.

    Steps:
      1. Read the image file from disk.
      2. Resize it to 224x224 (the input size MobileNetV2 expects).
      3. Convert color order from BGR to RGB (OpenCV loads as BGR by
         default, but the model was trained on RGB images).
      4. Return raw float32 pixel values in the 0-255 range.

    IMPORTANT: we do NOT divide by 255 here. Our model has a
    tf.keras.layers.Rescaling(scale=1./127.5, offset=-1) layer built
    directly into it, which expects raw 0-255 input and does the scaling
    itself. If we normalized here too, the image would be scaled twice,
    which breaks the model's predictions.
    """

    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Resize to the fixed size the model was trained on.
    img_resized = cv2.resize(img, target_size)

    # OpenCV loads images as BGR (Blue-Green-Red) by default.
    # Our model was trained on RGB images, so we must swap the channel order.
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)

    # Cast to float32. Values stay in the 0-255 range on purpose (see note above).
    img_float = img_rgb.astype(np.float32)

    return img_float

if __name__ == "__main__":
    print("preprocessing.py loaded successfully. No errors.")