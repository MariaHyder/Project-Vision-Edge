"""
fine_tune.py
------------
OPTIONAL, run this AFTER train.py if you want to try to push accuracy
higher. Unfreezes the top layers of MobileNetV2 and continues training
at a very low learning rate.

INPUT:  models/defect_model.keras (must already exist - run train.py first)
OUTPUT: models/defect_model.keras (overwritten with the fine-tuned version)

NOTE: Fine-tuning does not always help. It can improve overall accuracy
while making a specific type of mistake worse (or vice versa). ALWAYS
run evaluate.py and threshold_analysis.py before and after fine-tuning
and compare the numbers before deciding which version to keep.
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
FINE_TUNE_EPOCHS = 5
UNFREEZE_LAST_N_LAYERS = 30    # How many top MobileNetV2 layers become trainable
CACHE_DIR = 'cache'

# If your evaluate.py results show more "missed defects" than "false alarms",
# raise the weight on class 0 (defective) below to push the model to be more
# cautious about calling something "good".
CLASS_WEIGHT = {0: 1.5, 1: 1.0}


def load_datasets():
    train_dataset = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, shuffle=True, batch_size=BATCH_SIZE,
        image_size=IMG_SIZE, label_mode='binary'
    )
    test_dataset = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR, shuffle=False, batch_size=BATCH_SIZE,
        image_size=IMG_SIZE, label_mode='binary'
    )

    os.makedirs(CACHE_DIR, exist_ok=True)
    train_dataset = train_dataset.cache(os.path.join(CACHE_DIR, 'train_cache_ft'))
    test_dataset = test_dataset.cache(os.path.join(CACHE_DIR, 'test_cache_ft'))

    AUTOTUNE = tf.data.AUTOTUNE
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)
    return train_dataset, test_dataset


def main():
    print("Loading existing trained model...")
    model = tf.keras.models.load_model('models/defect_model.keras')

    # The MobileNetV2 base is nested inside our model as a single layer.
    # We search for it by name rather than assuming its position, since
    # that's more robust if the architecture changes slightly later.
    base_model = None
    for layer in model.layers:
        if 'mobilenet' in layer.name.lower():
            base_model = layer
            break
    if base_model is None:
        raise ValueError("Could not find the MobileNetV2 base layer inside the model.")

    print(f"Unfreezing the last {UNFREEZE_LAST_N_LAYERS} layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-UNFREEZE_LAST_N_LAYERS]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    train_data, test_data = load_datasets()

    print(f"\nFine-tuning for {FINE_TUNE_EPOCHS} epochs with class_weight={CLASS_WEIGHT}...")
    
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'models/defect_model.keras', 
        monitor='val_loss', 
        save_best_only=True
    )

       model.fit(
        train_data,
        validation_data=test_data,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=CLASS_WEIGHT,
        callbacks=[checkpoint]
    )

    print("Now run evaluate.py and threshold_analysis.py again to compare results.")

if __name__ == "__main__":
    main()