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
import logging as log

# logging
log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
log.disable(level=log.DEBUG)

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

    log.info(f'Reading MOS from {path}')
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

def easy_to_df(pred):
    """
    Converts the easyocr prediction to a dataframe.
    Parameters
    ----------
    pred : tuple
        The easyocr prediction.
    Returns
    -------
    pd.DataFrame
        The prediction as dataframe.
    """

    left = []
    top = []
    right = []
    bottom = []
    text = []
    conf = []

    for p in pred:
        left.append(p[0][0][0])
        top.append(p[0][0][1])
        right.append(p[0][1][0])
        bottom.append(p[0][2][1])
        text.append(p[1])
        conf.append(p[2])

    df = pd.DataFrame({'left': left, 'top': top, 'right': right, 'bottom': bottom, 'text': text, 'conf': conf})
    log.debug(f'easyocr prediction: {df}')
    df = df.round({'left': 0, 'top': 0, 'right': 0, 'bottom': 0, 'conf': 2})
    df = df.astype({'left': 'int', 'top': 'int', 'right': 'int', 'bottom': 'int'})

    return df

def pred_easy(img_paths):
    """
    Predicts the text in images using easyocr.
    Parameters
    ----------
    img_path : list(str)
        The paths to the images.
    Returns
    -------
    list
        The predicted texts as list of dataframes.
    """

    # load image
    results = []
    log.info('Loading easyocr reader')
    reader = easyocr.Reader(['en'])
    log.info('Reader loaded')
    for img_path in img_paths:
        pred = reader.readtext(img_path)

        # convert to dataframe
        pred = easy_to_df(pred)
        results.append(pred)

    return results


def tess_trans_df(pred):

    pred = pred[~pred['text'].isna()]

    df = pd.DataFrame()
    df['left'] = pred['left']
    df['top'] = pred['top']
    df['right'] = pred['left'] + pred['width']
    df['bottom'] = pred['top'] + pred['height']
    df['text'] = pred['text']
    df['conf'] = pred['conf']/100

    return df

def pred_tess(img_paths):
    """
    Predicts the text in images using tesseract.
    Parameters
    ----------
    img_path : list(str)
        The paths to the images.
    Returns
    -------
    list
        The predicted texts as list of dataframes.
    """

    results = []
    for img_path in img_paths:
        # load image
        with Image.open(img_path) as img:
            pred = pytesseract.image_to_data(img, output_type=pytesseract.Output.DATAFRAME)
        # convert dataframe to standard format
        pred = tess_trans_df(pred)
        results.append(pred)

    
    return results


def pred_data(img_paths, algo='easyocr'):
    """
    Predicts the text in a list of images in data format
    Parameters
    ----------
    img_paths : list
        The list of image paths.
    algo : str
        The algorithm to use for prediction.
    Returns
    -------
    list
        list of dataframes with predictions
    """

    if type(img_paths) is str:
        img_paths = [img_paths]
        
    results = []
    # Read the images.
    if algo == 'easyocr':
        results = pred_easy(img_paths)
    elif algo == 'tess':
        results = pred_tess(img_paths)

    return results


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

    numbers = [1,3,4,5]
    paths = [f"data/raw/scid/ReferenceSCIs/SCI0{num}.bmp" for num in numbers]
    paths = "data/raw/scid/ReferenceSCIs/SCI01.bmp"
    pred_easy = pred_data(paths)
    log.info(pred_easy[0].head())
    pred_tess = pred_data(paths, algo='tess')

    log.info(pred_tess[0].head())
