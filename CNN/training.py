import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Function to prepare the dataset
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
data_dir = './'
images, labels = prepare_dataset(data_dir)

# Count the occurrences of each label
unique, counts = np.unique(labels, return_counts=True)
label_counts = dict(zip(unique, counts))

print(f"Label counts: {label_counts}")

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalize pixel values and convert to float32
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Define data augmentation layers
data_augmentation = keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# Create TensorFlow datasets
batch_size = 32

# Training dataset
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
train_dataset = train_dataset.shuffle(buffer_size=1024)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Testing dataset
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test))
test_dataset = test_dataset.batch(batch_size)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Function to create the model
def create_model(input_shape):
    model = keras.Sequential([
        keras.Input(shape=input_shape),  # Specify input shape here
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    return model

# Create and compile the model
model = create_model((224, 224, 3))
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Implement early stopping to prevent overfitting
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=20,
    validation_data=test_dataset,
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Save the model using the Keras native format
model.save('small_model4.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(f'\nTest accuracy: {test_acc}')
