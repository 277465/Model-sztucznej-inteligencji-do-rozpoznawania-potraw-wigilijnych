import tkinter as tk
from tkinter import filedialog, ttk, Canvas
from PIL import Image, ImageTk
from predictor import predict_dish, get_class_indices

# Klasa GUI
class DishPredictorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Polish Christmas Dish Predictor")
        self.root.geometry("700x600")
        self.root.configure(bg="#f4f4f4")  # Tło okna

        # Pobieramy etykiety klas
        self.class_indices = get_class_indices()

        # Stylizacja elementów
        style = ttk.Style()
        style.configure("TButton", font=("Arial", 12), padding=6)
        style.configure("TLabel", font=("Arial", 12), background="#f4f4f4", foreground="#333333")

        # Tytuł aplikacji
        title_label = ttk.Label(root, text="Polish Christmas Dish Predictor", font=("Helvetica", 20, "bold"))
        title_label.pack(pady=20)

        # Ramka na wybór obrazu i wyświetlanie
        self.frame = ttk.Frame(root, padding=20)
        self.frame.pack(pady=10)

        # Przycisk do wyboru obrazu
        self.btn_select = ttk.Button(self.frame, text="Wybierz obraz", command=self.load_image)
        self.btn_select.grid(row=0, column=0, pady=10)

        # Canvas do wyświetlania obrazu
        self.canvas = Canvas(self.frame, width=300, height=300, bg="#dcdcdc", highlightthickness=0)
        self.canvas.grid(row=1, column=0, pady=20)

        # Wynik przewidywania
        self.result_label = ttk.Label(self.frame, text="Przewidywana potrawa: ", font=("Arial", 14, "bold"))
        self.result_label.grid(row=2, column=0, pady=10)

        # Ścieżka do obrazu
        self.img_path = None

    def load_image(self):
        # Otwieranie okna dialogowego do wyboru pliku
        self.img_path = filedialog.askopenfilename(
            title="Wybierz obraz potrawy",
            filetypes=[("Image Files", "*.jpg;*.jpeg;*.png;*.bmp")]
        )

        if self.img_path:
            self.display_image()
            self.predict_image()

    def display_image(self):
        # Wyświetlanie obrazu na Canvas
        img = Image.open(self.img_path)
        img = img.resize((300, 300))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(150, 150, image=self.tk_img)

    def predict_image(self):
        try:
            # Przewidywanie klasy potrawy
            predicted_dish = predict_dish(self.img_path, self.class_indices)
            self.result_label.config(text=f"Przewidywana potrawa: {predicted_dish}", foreground="#007B3E")
        except Exception as e:
            self.result_label.config(text=f"Błąd: {str(e)}", foreground="red")
