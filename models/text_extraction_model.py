import cv2
import pytesseract
from PIL import Image


def preprocess_with_otsu(image_path):
    # Read the image
    image = cv2.imread(image_path)

    # Convert to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Otsu's thresholding
    _, processed_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Convert back to PIL format for OCR
    pil_image = Image.fromarray(processed_image)
    # pil_image.show()  # Show the processed image for verification

    return pil_image


def extract_text_with_otsu(image_path):
    # Preprocess the image using Otsu's thresholding
    processed_image = preprocess_with_otsu(image_path)

    # Run Tesseract OCR on the processed image
    extracted_text = pytesseract.image_to_string(processed_image)

    print(f"Extracted Text with Otsu's Preprocessing:\n{extracted_text}")
    return extracted_text



