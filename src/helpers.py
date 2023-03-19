# file for helper functions
import editdistance
import os
import pytesseract
import re
import pandas as pd
from PIL import Image

# i/o functions


def text_error_rate(label, prediction):
    """
    Computes the text error rate (TER) between two strings.
    Parameters
    ----------
    label : str
        The ground truth label.
    prediction : str
        The predicted label.
    Returns
    -------
    float
        The text error rate between the two strings.
    """
    # Compute the edit distance between the two strings.
    distance = editdistance.eval(label, prediction)

    # Compute the length of the longest string.
    length = max(len(label), len(prediction))

    # Return the edit distance divided by the length of the longest string.
    return distance / length


def get_filenames(directory):
    """
    Gets the filenames in a directory.
    Parameters
    ----------
    directory : str
        The directory to search.
    Returns
    -------
    list
        The filenames.
    """
    # Get the filenames in the given directory.
    filenames = os.listdir(directory)

    # Remove the file extensions.
    filenames = [filename.split('.')[0] for filename in filenames]

    # Return the filenames.
    return filenames


def read_mos(path):

    print(f'Reading MOS from {path}')
    with open(path, encoding='utf-16') as f:
        text = f.read()

    match = r'(SCI(\d\d))\s+(SCI(\d\d)_(\d)_(\d))\s+([\d\.]+)'
    result = re.findall(match, text)

    cols = ['ref', 'ref_num', 'img', 'img_num', 'comp', 'qual', 'mos']
    types = ['str', 'int', 'str', 'int', 'int', 'int', 'float']
    types = dict(zip(cols, types))

    df = pd.DataFrame(result, columns=cols)
    df = df.astype(types)

    return df


def pred_img(img_path, label_path):

    # load image
    with Image.open(img_path) as img:
        # run tesseract and save prediction
        pred = pytesseract.image_to_string(img)

    # img_name = img_path.split('/')[-1]
    # print(f'Ran tesseract on {img_name}')

    # load label, compare and save text-error-rate
    with open(label_path) as f:
        label = f.read()
    ter = text_error_rate(pred, label)

    # print(f'Calculated text error rate for {img_name}, TER: {ter}')

    return ter


if __name__ == '__main__':
    # Compute the text error rate between two strings.
    label = 'this is a test'
    prediction = 'this is a test!'
    print(text_error_rate(label, prediction))
