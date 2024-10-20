import os
from PIL import Image

def reconstruct_images(output_dir, values):
    images = []
    threshold = 0.50

    # Create a combined list of images with their values
    for filename, value in values.items():
        value = float(value)
        images.append((os.path.join(output_dir, filename), value))  # Store path and value as a tuple

    # Initialize counters for naming
    count = 1
    current_block = []
    current_type = None  # None, 'visual', or 'text'

    # Process images in order
    for img_path, value in images:
        if value > threshold:
            img_type = 'visual'
        else:
            img_type = 'text'

        # If the type changes, combine the current block
        if current_type is None:
            current_type = img_type

        if img_type != current_type:
            # Combine the current block
            if current_block:
                combined_image = combine_images(current_block)
                combined_image.save(os.path.join('result', f'{count}.png'))  # Save combined image
                count += 1
                current_block = []  # Reset the current block

            current_type = img_type  # Update the current type

        current_block.append(img_path)  # Add the current image to the block

    # Combine any remaining images in the last block
    if current_block:
        combined_image = combine_images(current_block)
        combined_image.save(os.path.join('result', f'{count}.png'))  # Save combined image

    print("Reconstruction complete. Check the 'result' folder for output.")

def combine_images(image_paths):
    combined_image = None
    y_offset = 0
    images_to_combine = []

    for img_path in image_paths:
        img = Image.open(img_path)

        # If combined_image is None, initialize it with the first image's dimensions
        if combined_image is None:
            combined_image = Image.new('RGB', (img.width, 0))  # Start with the width of the first image

        # Add the image to the list for combining
        images_to_combine.append(img)
        combined_image = combined_image.resize((img.width, combined_image.height + img.height))  # Update height

    # Paste images into the combined image
    for img in images_to_combine:
        combined_image.paste(img, (0, y_offset))  # Stack images vertically
        y_offset += img.height  # Update the y offset

    return combined_image
