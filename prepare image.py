from PIL import Image
import numpy as np


def prepare_image(img_path, target_size=(224, 224), min_size=(100, 100)):
    """
    Przygotowuje obraz do predykcji:
    - Sprawdza, czy rozmiar obrazu jest większy niż min_size.
    - Skaluje obraz do target_size, zachowując proporcje.
    """
    try:
        with Image.open(img_path) as img:
            # Sprawdzenie minimalnych rozmiarów obrazu
            if img.width < min_size[0] or img.height < min_size[1]:
                raise ValueError(
                    "Obraz jest za mały. Minimalne wymiary to 100x100 pikseli.")

            # Konwersja na format RGB (jeśli np. WebP lub inne formaty)
            img = img.convert("RGB")

            # Skalowanie i dopasowanie proporcji
            img.thumbnail(target_size,
                          Image.ANTIALIAS)  # Zmniejszenie obrazu z zachowaniem proporcji

            # Uzupełnianie do target_size (224x224) przez dodanie marginesów
            background = Image.new("RGB", target_size,
                                   (255, 255, 255))  # Białe tło
            x_offset = (target_size[0] - img.width) // 2
            y_offset = (target_size[1] - img.height) // 2
            background.paste(img, (x_offset, y_offset))

            # Konwersja na tablicę NumPy i normalizacja
            img_array = np.array(background) / 255.0
            img_array = np.expand_dims(img_array,
                                       axis=0)  # Dodanie wymiaru batch
            return img_array
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None
