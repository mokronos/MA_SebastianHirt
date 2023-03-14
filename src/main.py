# load data
# define paths
from PIL import Image
import helpers
import pytesseract
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# split functions into i/o and processing
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
    ter = helpers.text_error_rate(pred, label)

    # print(f'Calculated text error rate for {img_name}, TER: {ter}')

    return ter


reference_dir = 'data/raw/ReferenceSCIs'
compressed_dir = 'data/raw/DistortedSCIs'
mos_path = 'data/raw/MOS_SCID.txt'
labels_dir = 'labels/raw'

image_ext = '.bmp'
label_ext = '.txt'
prefix = 'SCI'

# get filenames
numbers = list(range(1, 41))
# print(numbers)
# numbers = [4, 5, 6, 22, 29]
# numbers = numbers[:2]
numbers = [x if x > 9 else f'0{x}' for x in numbers]
compressions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
qualities = [1, 2, 3, 4, 5]

data = read_mos(mos_path)
data['ter'] = np.nan

for num in numbers:
    for comp in compressions:
        for qual in qualities:
            # just text, ignore bounding box/position
            # run on different compression levels (SCID dataset)
            img_name = f'{prefix}{num}_{comp}_{qual}{image_ext}'
            print(f'Running for {img_name}...')
            label_name = f'{prefix}{num}{label_ext}'
            img_path = f'{compressed_dir}/{img_name}'
            label_path = f'{labels_dir}/{label_name}'

            ter = pred_img(img_path, label_path)
            data.loc[(data.img_num == int(num))
                     & (data.comp == comp)
                     & (data.qual == qual), 'ter'] = ter

# compare to MOS of dataset, somehow
data['ter_comp'] = (1 - data['ter']) * 100
data.to_csv('results/ter.csv', index=False)
data.dropna(inplace=True)

# plot
# data.plot.scatter(x='mos', y='ter_comp')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.grid()
# plt.show()
