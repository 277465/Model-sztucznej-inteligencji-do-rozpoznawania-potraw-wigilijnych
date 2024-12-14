import tensorflow as tf
import numpy as np
from tkinter import filedialog, ttk, Canvas
from PIL import Image, ImageTk

IMAGE_SIZE = (224, 224)
MODEL_SAVE_PATH = 'polish_christmas_model.keras'
DATASET_DIR = 'dataset'

model = tf.keras.models.load_model(MODEL_SAVE_PATH)

class_indices = {
'barszcz czerwony':0,
'bigos':1,
'kutia':2,
'makowiec':3,
'pierniki':4,
'pierogi':5,
'sernik':6,
'zupa grzybowa':7}
def prepare_image(img_path, target_size=(224, 224), min_size=(100, 100)):
    try:
        with Image.open(img_path) as img:
            if img.width < min_size[0] or img.height < min_size[1]:
                raise ValueError(
                    "Obraz jest za mały. Minimalne wymiary to 100x100 pikseli.")

            img = img.convert("RGB")

            img.thumbnail(target_size, Image.Resampling.LANCZOS)

            background = Image.new("RGB", target_size,
                                   (255, 255, 255))  # Białe tło
            x_offset = (target_size[0] - img.width) // 2
            y_offset = (target_size[1] - img.height) // 2
            background.paste(img, (x_offset, y_offset))


            img_array = np.array(background) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            return img_array
    except Exception as e:
        print(f"Błąd podczas przetwarzania obrazu: {e}")
        return None


def predict_dish(img_path, class_indices, threshold=0.79):
    img = prepare_image(img_path)
    if img is None:
        return "Błąd w przygotowaniu obrazu."

    predictions = model.predict(img)
    predicted_probabilities = predictions[0]
    predicted_class_index = np.argmax(predicted_probabilities)
    predicted_class_label = {v: k for k, v in class_indices.items()}[predicted_class_index]
    predicted_probability = predicted_probabilities[predicted_class_index]

    if predicted_probability < threshold:
        return "Na zdjęciu nie ma rozpoznawalnego dania świątecznego."

    return f"Rozpoznane danie: {predicted_class_label}"


class DishPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polish Christmas Dish Predictor")
        self.root.geometry("700x600")
        self.root.configure(bg="#f4f4f4")

        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=6)
        style.configure("TLabel", font=("Arial", 12), background="#f4f4f4",
                        foreground="#333333")

        title_label = ttk.Label(root, text="Polish Christmas Dish Predictor",
                                font=("Helvetica", 20, "bold"))
        title_label.pack(pady=20)

        self.frame = ttk.Frame(root, padding=20)
        self.frame.pack(pady=10)

        self.btn_select = ttk.Button(self.frame, text="Wybierz obraz",
                                     command=self.load_image)
        self.btn_select.grid(row=0, column=0, pady=10)

        self.canvas = Canvas(self.frame, width=300, height=300, bg="#dcdcdc",
                             highlightthickness=0)
        self.canvas.grid(row=1, column=0, pady=20)

        self.result_label = ttk.Label(self.frame,
                                      text="Przewidywana potrawa: ",
                                      font=("Arial", 14, "bold"))
        self.result_label.grid(row=2, column=0, pady=10)

        self.img_path = None

    def load_image(self):
        self.img_path = filedialog.askopenfilename(
            title="Wybierz obraz potrawy",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.webp")]
        )

        if self.img_path:
            self.display_image()
            self.predict_image()

    def display_image(self):
        img = Image.open(self.img_path)
        img = img.resize((300, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(150, 150, image=self.tk_img)

    def predict_image(self):
        try:
            predicted_dish = predict_dish(self.img_path, class_indices)
            self.result_label.config(
                text=f"Przewidywana potrawa: {predicted_dish}",
                foreground="#007B3E")
        except Exception as e:
            self.result_label.config(text=f"Błąd: {str(e)}", foreground="red")
