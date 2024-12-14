import tensorflow as tf
import numpy as np
from PIL import Image

# Ustawienia
MODEL_SAVE_PATH = 'polish_christmas_model.keras'
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Mapowanie klas (klasyfikacja według dostarczonych etykiet)
class_indices = {
    'barszcz czerwony': 0,
    'bigos': 1,
    'kutia': 2,
    'makowiec': 3,
    'pierniki': 4,
    'pierogi': 5,
    'sernik': 6,
    'zupa grzybowa': 7
}

# Załaduj model
model = tf.keras.models.load_model(MODEL_SAVE_PATH)

# Funkcja przygotowania obrazu
def prepare_image(img_path, target_size=(224, 224), min_size=(100, 100)):
    try:
        with Image.open(img_path) as img:
            if img.width < min_size[0] or img.height < min_size[1]:
                raise ValueError("Obraz jest za mały. Minimalne wymiary to 100x100 pikseli.")
            img = img.convert("RGB")
            img.thumbnail(target_size, Image.Resampling.LANCZOS)

            background = Image.new("RGB", target_size, (255, 255, 255))  # Białe tło
            x_offset = (target_size[0] - img.width) // 2
            y_offset = (target_size[1] - img.height) // 2
            background.paste(img, (x_offset, y_offset))

            img_array = np.array(background) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None

# Funkcja przewidywania
def predict_dish(img_path, class_indices, threshold=0.79):
    img = prepare_image(img_path)
    if img is None:
        return "Błąd w przygotowaniu obrazu."

    predictions = model.predict(img)
    predicted_probabilities = predictions[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = {v: k for k, v in class_indices.items()}[predicted_class_index]
    predicted_probability = predicted_probabilities[predicted_class_index]

    # Sprawdzanie progu
    if predicted_probability < threshold:
        return "Na zdjęciu nie ma rozpoznawalnego dania świątecznego."

    return f"Rozpoznane danie: {predicted_class_label}"

# Przykład użycia:
# img_path = 'path_to_image.jpg'
# result = predict_dish(img_path, class_indices)
# print(result)

