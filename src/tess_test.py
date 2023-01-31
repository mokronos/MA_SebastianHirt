# path stuff
import os
# to load images
from PIL import Image
# tesseract libary to read text from image
import pytesseract as tess


def img_to_text(image_name, output_name):
    """
    Reads text from image and writes it to a file.
    """

    # get current path
    current_path = os.path.dirname(__file__)
    # get parent path
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))

    # define image path
    image_path = os.path.join(parent_path, image_name)
    # define output path
    output_path = os.path.join(parent_path, output_name)

    # load image
    image = Image.open(image_path)

    # read text from image
    text = tess.image_to_string(image)

    # write text to file
    with open(output_path, 'w') as f:
        f.write(text)


# sample image
image_name = "sampleimage.png"
# define output name
output_name = "sampleoutput.txt"

# run function
img_to_text(image_name, output_name)

# sample image from dataset
image_path2 = "data/DistortedSCIs/SCI01_1_1.bmp"
# image_path2 = "SCI01_1_1.bmp"
# define output name
output_name2 = "datasampleoutput.txt"

# run function
img_to_text(image_path2, output_name2)
