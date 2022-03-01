# @description: Implementation of OCR using tesseract
# @author: Zzay
# @create: 2022/03/01 13:53
import pytesseract
from PIL import Image

# Load data
filepath = '../res/img/text-img.png'

# Recognize characters
text = pytesseract.image_to_string(
    image=Image.open(filepath),  # Image to be scanned
    lang='chi_sim')  # Language of text expected

# Print text recognized
print(text)
