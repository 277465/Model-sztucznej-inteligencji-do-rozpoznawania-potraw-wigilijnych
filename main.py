import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# --- 1. Ładowanie etykiet ---
def load_labels(label_file):
    label_dict = {}
    with open(label_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()[1:]  # Pomijamy nagłówek
        for line in lines:
            label_name, label_id = line.strip().split(',')
            label_dict[label_name] = int(label_id)
    return label_dict

label_file = 'labels.txt'
labels = load_labels(label_file)
print("Załadowane etykiety:", labels)

# --- 2. Parametry danych ---
IMG_SIZE = 224  # Rozmiar obrazu (do zmiany na potrzeby modelu)
BATCH_SIZE = 32
EPOCHS = 10  # Liczba epok treningowych

# --- 3. Przygotowanie danych treningowych i walidacyjnych ---
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizacja pikseli do zakresu [0,1]
    validation_split=0.2  # Podział na zbiór treningowy i walidacyjny
)

train_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    color_mode='rgb'
)

val_generator = datagen.flow_from_directory(
    'dataset',
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    color_mode='rgb'
)

# --- 4. Budowa modelu CNN ---
def create_model(num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        MaxPooling2D(2, 2),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    return model

# Tworzenie modelu
model = create_model(num_classes=len(labels))
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# --- 5. Trenowanie modelu ---
print("Rozpoczynanie treningu modelu...")
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# Zapisanie wytrenowanego modelu
model.save('food_classifier_model.h5')
print("Model zapisany jako 'food_classifier_model.h5'.")

# --- 6. Testowanie modelu na pojedynczym obrazie ---
def predict_image(model, image_path):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0  # Normalizacja
    img = np.expand_dims(img, axis=0)  # Dodanie wymiaru batch_size
    
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    class_labels = list(train_generator.class_indices.keys())  # Pobranie etykiet klas
    print(f"Przewidywana klasa dla obrazu '{image_path}': {class_labels[predicted_class[0]]}")

# Przykład użycia funkcji predykcji
test_image_path = 'test_images/test1.jpg'  # Upewnij się, że ścieżka jest poprawna
if os.path.exists(test_image_path):
    predict_image(model, test_image_path)
else:
    print(f"Plik testowy '{test_image_path}' nie istnieje.")
