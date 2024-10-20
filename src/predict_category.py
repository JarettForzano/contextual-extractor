import os
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array

def load_and_preprocess_images(image_folder):
    images = []
    filenames = []
    
    # Load images and their filenames
    for filename in sorted(os.listdir(image_folder), key=lambda x: int(x.split('.')[0])):
        if filename.endswith('.png'):
            img_path = os.path.join(image_folder, filename)
            img = load_img(img_path, target_size=(224, 224))  # Adjust size as needed
            img_array = img_to_array(img) / 255.0  # Normalize the image
            images.append(img_array)
            filenames.append(filename)
    
    return np.array(images), filenames

def predict_with_model(model_path, images):
    model = load_model(model_path)
    predictions = model.predict(images)
    return predictions