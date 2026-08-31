"""
train.py
--------
Trains the defect-detection model from scratch using transfer learning
on MobileNetV2. Run this FIRST, before any other script.

INPUT:  images in data/train/<class_name>/ and data/test/<class_name>/
OUTPUT: models/defect_model.keras  (the trained Keras model)

Two improvements over a "bare minimum" training script are included here,
both explained inline where they appear:
  1. Disk caching  -> makes epochs 2 and onward much faster.
  2. Data augmentation -> makes the model less sensitive to lighting,
     angle, and camera noise differences (helps with real-world/webcam use).
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'      
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'     

import silence_tensorflow.auto   
import tensorflow as tf        

TRAIN_DIR = 'data/train'
TEST_DIR = 'data/test'
BATCH_SIZE = 32
IMG_SIZE = (224, 224)
EPOCHS = 5
CACHE_DIR = 'cache'   


def load_datasets():
    """
    Loads images from disk into tf.data.Dataset objects, which TensorFlow
    can efficiently feed into the model during training.
    """
    print("Loading training data...")
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        shuffle=True,              # Shuffle image order every epoch (important for training)
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,       # Automatically resizes every image to 224x224
        label_mode='binary'        # Two classes -> single 0/1 label per image
    )

    print("Loading testing data...")
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        shuffle=False,           
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        label_mode='binary'
    )

    os.makedirs(CACHE_DIR, exist_ok=True)

    # -----------------------------------------------------------------
    # WHY TRAINING TAKES SO LONG, AND HOW .cache() FIXES PART OF IT:
    # Every epoch, TensorFlow re-reads every JPEG from disk, decodes it,
    # and resizes it — that decode/resize work is repeated identically
    # 5 times (once per epoch) even though the images never change.
    # .cache() saves the decoded, resized images to disk the FIRST time
    # through, so epochs 2-5 skip straight to "already decoded" data.
    # Epoch 1 will NOT be faster (the cache is still being built), but
    # epochs 2 onward should be noticeably quicker.
    # -----------------------------------------------------------------
    train_dataset = train_dataset.cache(os.path.join(CACHE_DIR, 'train_cache'))
    test_dataset = test_dataset.cache(os.path.join(CACHE_DIR, 'test_cache'))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

    return train_dataset, test_dataset


def build_mobilenetv2_model(input_shape=(224, 224, 3)):
    """
    Builds the model architecture: MobileNetV2 (pretrained on ImageNet) as
    a frozen feature extractor, with a small trainable classifier head on top.
    """

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )

    # Freeze the base model - its ImageNet-learned weights stay fixed, and
    # only our new classifier head layers get trained. This is standard
    # "transfer learning" and trains much faster than training from scratch.
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)

    # -----------------------------------------------------------------
    # DATA AUGMENTATION LAYERS
    # These randomly flip, rotate, brighten, contrast-shift, and zoom the
    # TRAINING images only. Keras automatically disables them during
    # validation/testing and during real inference (training=False), so
    # they never affect infer.py, evaluate.py, or the live webcam demo -
    # they only make training see more varied examples, which helps the
    # model generalize to different lighting and camera angles later.
    # This does NOT make the model immune to all lighting/noise conditions,
    # but it meaningfully improves robustness compared to no augmentation.
    # -----------------------------------------------------------------
    x = tf.keras.layers.RandomFlip('horizontal')(inputs)
    x = tf.keras.layers.RandomRotation(0.05)(x)
    x = tf.keras.layers.RandomBrightness(0.2)(x)
    x = tf.keras.layers.RandomContrast(0.2)(x)
    x = tf.keras.layers.RandomZoom(0.1)(x)

    # Rescaling layer: converts raw 0-255 pixel values into the -1 to 1
    # range MobileNetV2 expects. This is baked into the model itself so
    # preprocessing.py never has to do this step manually.
    x = tf.keras.layers.Rescaling(scale=1./127.5, offset=-1)(x)

    # Run the (frozen) MobileNetV2 feature extractor.
    x = base_model(x, training=False)

    # Turn the spatial feature map into a single flat vector per image.
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dropout randomly disables 20% of neurons during training only,
    # which helps prevent overfitting.
    x = tf.keras.layers.Dropout(0.2)(x)

    # Final output: one number between 0 and 1 (sigmoid), our defect score.
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='binary_crossentropy',   # Standard loss for 2-class problems
        metrics=['accuracy']
    )

    return model


if __name__ == "__main__":
    train_data, test_data = load_datasets()
    model = build_mobilenetv2_model()

    print("\nModel built successfully! Starting training phase...")
    print(f"Training for {EPOCHS} epochs. Epoch 1 will be slower (building the")
    print("disk cache); epochs after that should speed up noticeably.\n")

    history = model.fit(
        train_data,
        validation_data=test_data,
        epochs=EPOCHS
    )

    os.makedirs('models', exist_ok=True)
    model.save('models/defect_model.keras')
    print("\nTraining complete! Model saved to 'models/defect_model.keras'.")