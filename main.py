from slicing import slice_pdf_to_images
from dotenv import load_dotenv

load_dotenv()


def main():
    input_pdf = "convert/TCP_IP.pdf"
    output_dir = "photos"
    slice_height = 150
    slice_pdf_to_images(input_pdf, output_dir, slice_height)
    
if __name__ == "__main__":
    main()

