import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

model = tf.keras.models.load_model(MODEL_SAVE_PATH)

def predict_dish(img_path, class_indices, threshold=0.79):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    predicted_probabilities = predictions[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = {v: k for k, v in class_indices.items()}[predicted_class_index]
    predicted_probability = predicted_probabilities[predicted_class_index]

    if predicted_probability < threshold:
        return "Na zdjęciu nie ma rozpoznawalnego dania świątecznego."

    return f"Rozpoznane danie: {predicted_class_label}"

def get_class_indices():
    train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
    train_generator = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMAGE_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    return train_generator.class_indices
