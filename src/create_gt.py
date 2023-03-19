import glob
from PIL import Image
import pytesseract as tess


data_path = 'data/raw/ReferenceSCIs/'

labels_path = 'labels/exp/'


# get all the files in the data folder
image_paths = glob.glob(data_path + '*.bmp')


def label_image(image_path):

    # get the image name
    image_name = image_path.split('/')[-1].split('.')[0]

    # get label with tesseract
    with Image.open(image_path) as img:
        label = tess.image_to_string(img, config='--psm 11 --oem 1')

    # write text to file
    with open(labels_path + image_name + ".txt", 'w') as f:
        f.write(label)

    return


image_paths.sort()
image_paths = image_paths[:5]
for image_path in image_paths:

    print(f"Labeling {image_path}")
    label_image(image_path)
