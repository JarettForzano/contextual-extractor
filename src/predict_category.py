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
            
            # Check for blank slice by counting non-white pixels
            non_white_pixels = np.sum(img_array < 0.95)
            total_pixels = img_array.size  # Total number of pixels in the image
            
            # If more than a small fraction of pixels are non-white, consider it non-blank
            if non_white_pixels / total_pixels > 0.01:  # Adjust the threshold as needed
                images.append(img_array)
                filenames.append(filename)
    
    return np.array(images), filenames

def predict_with_model(model_path, images):
    model = load_model(model_path)
    predictions = model.predict(images)
    return predictions
