from PyPDF2 import PdfReader, PdfWriter
from PyPDF2.generic import RectangleObject
from pathlib import Path
import fitz

def slice_pdf(input_pdf, output_prefix, slice_heights):
    reader = PdfReader(input_pdf)
    page = reader.pages[0]
    
    # Get the original page dimensions
    original_width = float(page.mediabox.width)
    original_height = float(page.mediabox.height)
    
    current_y = original_height
    for i, height in enumerate(slice_heights):
        writer = PdfWriter()
        new_page = reader.pages[0].copy()
        
        # Calculate the crop box
        crop_bottom = max(0, current_y - height)
        crop_box = RectangleObject([0, crop_bottom, original_width, current_y])
        
        # Set the crop box directly
        new_page["/CropBox"] = crop_box
        
        writer.add_page(new_page)
        
        with open(f"{output_prefix}_{i+1}.pdf", "wb") as output_file:
            writer.write(output_file)
        
        current_y -= height

def slice_pdf_to_images(input_pdf, output_dir, slice_height):
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    doc = fitz.open(input_pdf)
    slice_number = 1
    for page_number in range(len(doc)):
        page = doc[page_number]
        original_width, original_height = page.rect.width, page.rect.height
        current_y = 0
        
        while current_y < original_height:
            height = slice_height
            if current_y + height > original_height:
                height = original_height - current_y
            rect = fitz.Rect(0, current_y, original_width, current_y + height)
            pix = page.get_pixmap(clip=rect)
            output_path = Path(output_dir) / f"p5_{slice_number}.png"
            pix.save(output_path)
            current_y += height
            slice_number += 1

