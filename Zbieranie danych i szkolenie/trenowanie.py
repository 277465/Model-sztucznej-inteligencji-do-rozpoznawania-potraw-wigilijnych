import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Set parameters
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 200
NUM_CLASSES = 8  # Number of classes, update if necessary
DATASET_DIR = 'dataset'  # Path to your dataset directory
MODEL_SAVE_PATH = 'polish_christmas_model.keras'

# 1. Data Preparation with Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80-20 split for training and validation
    rotation_range=45,  # Increased rotation range
    width_shift_range=0.3,  # Increased width shift
    height_shift_range=0.3,  # Increased height shift
    shear_range=0.3,  # Increased shear range
    zoom_range=0.3,   # Increased zoom range
    horizontal_flip=True,
    fill_mode='nearest'
)

train_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

validation_generator = train_datagen.flow_from_directory(
    DATASET_DIR,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=True
)

# 2. Build the Model Using Transfer Learning (with Fine-Tuning)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=IMAGE_SIZE + (3,))
base_model.trainable = False  # Freeze the base model initially

# Add custom layers on top of the pre-trained base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),  # Added BatchNormalization for better training stability
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(NUM_CLASSES, activation='softmax')
])

# Fine-tuning the base model after some initial training
base_model.trainable = True
for layer in base_model.layers[:-20]:  # Unfreeze the last few layers of the base model
    layer.trainable = False

# 3. Compile the Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Use a smaller learning rate
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 4. Set Up Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ModelCheckpoint(MODEL_SAVE_PATH, save_best_only=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)  # Reduce learning rate on plateau
]

# 5. Train the Model
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

# 6. Save the Model
model.save(MODEL_SAVE_PATH)

print("Model training complete and saved as 'polish_christmas_model.keras'.")
print(f"Class indices: {train_generator.class_indices}")
