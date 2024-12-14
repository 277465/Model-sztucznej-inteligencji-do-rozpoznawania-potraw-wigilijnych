import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Konfiguracja modelu
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'polish_christmas_model.keras'

# Wczytanie modelu
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Funkcja przygotowania obrazu
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=IMAGE_SIZE)
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Tworzy wymiar wsadowy (batch)
    img_array /= 255.0  # Skalowanie wartości pikseli
    return img_array

# Funkcja do przewidywania klasy obrazu
def predict_dish(img_path, class_indices, threshold=0.79):
    """
    Przewiduje, czy obraz zawiera świąteczne danie.
    Jeśli nie, zwraca informację o braku odpowiedniego dania.
    """
    img = prepare_image(img_path)  # Przygotowanie obrazu
    predictions = model.predict(img)  # Predykcja modelu

    # Pobieramy prawdopodobieństwa dla wszystkich klas
    predicted_probabilities = predictions[0]
    predicted_class_index = np.argmax(predicted_probabilities)  # Indeks klasy o największym prawdopodobieństwie
    predicted_class_label = {v: k for k, v in class_indices.items()}[predicted_class_index]
    predicted_probability = predicted_probabilities[predicted_class_index]

    # Jeśli prawdopodobieństwo jest mniejsze niż próg, uznajemy, że nie ma dania
    if predicted_probability < threshold:
        return "Na zdjęciu nie ma rozpoznawalnego dania świątecznego."

    # Zwracamy rozpoznane danie
    return f"Rozpoznane danie: {predicted_class_label} "

# Generator klas
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
