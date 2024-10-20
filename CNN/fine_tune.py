import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras import layers
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from PIL import Image
from keras.callbacks import EarlyStopping

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

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    images, labels, test_size=0.2, random_state=42
)

# Normalize pixel values and convert to float32
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0

# Load the MobileNetV2 model
base_model = keras.applications.MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')
base_model.trainable = False  # Freeze the base model

# Create the model
model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(1, activation='sigmoid')  # Adjust output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Compute class weights to handle class imbalance
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train),
    y=y_train
)
class_weights_dict = dict(enumerate(class_weights))

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
history = model.fit(
    X_train, y_train,
    epochs=20,  # Start with 20 epochs
    validation_data=(X_test, y_test),
    class_weight=class_weights_dict,
    callbacks=[early_stopping]
)

# Save the model
model.save('fine_tuned_mobilenet.keras')

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'\nTest accuracy: {test_acc}')
