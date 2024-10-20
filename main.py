from src.slicing import slice_pdf_to_images
from dotenv import load_dotenv
from src.predict_category import load_and_preprocess_images, predict_with_model
from src.reconstruct import reconstruct_images
import glob
import os
load_dotenv()

def clear_photos_directory(directory):
    for file in glob.glob(os.path.join(directory, '*')):
        os.remove(file)

def main():
    
    input_pdf = "convert/Training Model.pdf"
    output_dir = "photos"
    model_path = "models/fine_tuned_mobilenet.keras"
    clear_photos_directory(output_dir)

    slice_height = 150
    slice_pdf_to_images(input_pdf, output_dir, slice_height)
    
    images, filenames = load_and_preprocess_images(output_dir)
    predictions = predict_with_model(model_path, images)

    results = {filename: prediction for filename, prediction in zip(filenames, predictions)}
    #print(results)

    reconstruct_images(output_dir, results)
    # When we fine tuned the model we set it so that if above 0.5 it is visual and below it is text
    clear_photos_directory(output_dir)
    

if __name__ == "__main__":
    main()
