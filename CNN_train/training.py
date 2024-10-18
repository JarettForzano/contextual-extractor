import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def prepare_dataset(data_dir, image_size=(224, 224)):
    images = []
    labels = []
    
    for category in ['visual', 'text']:
        path = os.path.join(data_dir, category)
        class_num = 1 if category == 'visual' else 0
        for img in os.listdir(path):
            try:
                img_path = os.path.join(path, img)
                img_array = Image.open(img_path).convert('RGB')
                img_array = img_array.resize(image_size)
                images.append(np.array(img_array))
                labels.append(class_num)
            except Exception as e:
                print(f"Error loading image {img}: {e}")
    
    return np.array(images), np.array(labels)

# Prepare the dataset
data_dir = 'path/to/your/dataset'  # Replace with your dataset path
images, labels = prepare_dataset(data_dir)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalize pixel values
X_train = X_train.astype('float32') / 255
X_test = X_test.astype('float32') / 255

def create_model(input_shape):
    model = keras.Sequential([
        keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.MaxPooling2D((2, 2)),
        keras.layers.Conv2D(64, (3, 3), activation='relu'),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile the model
model = create_model((224, 224, 3))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Data augmentation for training
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

# Train the model
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train) // 32,
    epochs=20,
    validation_data=(X_test, y_test)
)

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')

# Function to predict on new images
def predict_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Visual" if prediction[0][0] > 0.5 else "Text"

# # Example usage
# new_image_path = 'path/to/new/image.png'
# result = predict_image(new_image_path, model)
# print(f"The image is predicted to be: {result}")
