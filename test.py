import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20
NUM_CLASSES = 8
DATASET_DIR = 'dataset'
MODEL_SAVE_PATH = 'polish_christmas_model.keras'


train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,  # 80-20 split for training and validation
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
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


# Load the model
model = tf.keras.models.load_model('polish_christmas_model.keras')

# Function to prepare the image
def prepare_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)  # Create batch axis
    img_array /= 255.0  # Rescale
    return img_array

# Prediction function
def predict_dish(img_path, class_indices):
    img = prepare_image(img_path)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)
    labels = {v: k for k, v in class_indices.items()}
    return labels[predicted_class[0]]


# Example usage
if __name__ == "__main__":
    test_image_path = './examples/zupa grzybowa.jpg'  # Replace with your image path
    class_indices = train_generator.class_indices
    dish = predict_dish(test_image_path, class_indices)
    print(f"The dish is: {dish}")
