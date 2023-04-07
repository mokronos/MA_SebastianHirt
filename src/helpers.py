# file for helper functions
import editdistance
import os
import pytesseract
import easyocr
import re
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit
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


def pred_img(img_path, label_path, algo='easyocr'):

    # load image
    pred = ''
    if algo == 'ezocr':
        # lead easyocr
        # might make sense to init object once pass list to function
        reader = easyocr.Reader(['en'])
        pred = reader.readtext(img_path, detail=0, paragraph=True)
    elif algo == 'tess':
        with Image.open(img_path) as img:
            # run tesseract and save prediction
            pred = pytesseract.image_to_string(img)
            pred = pred.splitlines()

    return pred


def nonlinearfitting(objvals, subjvals, max_nfev=400):
    """
    code adapted from:
    https://github.com/lllllllllllll-llll/SROCC_PLCC_calculate/blob/master/nonlinearfitting.m
    """

    # calculate SROCC before the non-linear mapping
    srocc, _ = spearmanr(objvals, subjvals)

    # define the nonlinear fitting function
    # found in this paper:
    # https://arxiv.org/pdf/1810.08169.pdf
    # originally from this:
    # https://www.researchgate.net/publication/221458323_Video_Quality_Experts_Group_current_results_and_future_directions
    # humans might be less sensitive to the difference between images
    def model(x, a, b, c, d):
        return ((a-b)/(1+np.exp((-x+c)/d)))+b

    # initialize the parameters used by the nonlinear fitting function
    beta0 = [np.max(subjvals), np.min(subjvals),
             np.mean(objvals), np.std(objvals)/4]

    # fitting a curve using the data
    betam, _ = curve_fit(model, objvals, subjvals, p0=beta0, method='lm',
                         maxfev=max_nfev)

    # given an objective value,
    # predict the corresponding MOS (ypre) using the fitted curve
    # ypre are the modified objective values, not subjective (90% sure)
    ypre = model(np.array(objvals), *betam)

    plcc, _ = pearsonr(subjvals, ypre)  # pearson linear coefficient
    return srocc, plcc, ypre


if __name__ == '__main__':
    # Compute the text error rate between two strings.
    label = 'this is a test'
    prediction = 'this is a test!'
    print(text_error_rate(label, prediction))
