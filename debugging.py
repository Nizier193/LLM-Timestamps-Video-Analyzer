from image_analysis import ImageAnalysis
from PIL import Image

ia = ImageAnalysis()

image = Image.open("pitch_1_output/pitch_1_00000.png")

def concantenate_images(image1, image2):
    pass

ia.analyze_image(image)
