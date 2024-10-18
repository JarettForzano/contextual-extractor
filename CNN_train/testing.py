import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model
model_path = 'small_model2.keras'
model = load_model(model_path)

def predict_image(image_path, model):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return "Visual" if prediction[0][0] > 0.5 else "Text"


new_image_path = 'visual/19.png'
result = predict_image(new_image_path, model)
print(f"The image is predicted to be: {result}")
