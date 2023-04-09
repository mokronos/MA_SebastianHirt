# path stuff
import logging as log
import pytesseract
import easyocr
from PIL import Image

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# log.disable(log.CRITICAL)

def tess_predict(img_paths):

    results = []
    for img_path in img_paths:
        # run tesseract and save prediction
        pred = pytesseract.image_to_string(Image.open(img_path))
        results.append(pred)

    return results

image_path = "data/raw/scid/ReferenceSCIs/SCI03.bmp"

tessdata = pytesseract.image_to_data(Image.open(image_path), lang='eng', output_type=pytesseract.Output.DATAFRAME)
tessstr = pytesseract.image_to_string(Image.open(image_path), lang='eng')

easydata = easyocr.Reader(['en']).readtext(image_path)

tessdata = tessdata[~tessdata['text'].isna()]

tesstext = tessdata['text'].to_list()
tessstr = ' '.join(tesstext)

log.debug(f'easyocr result: {easydata}')

easytext = [line[1] for line in easydata]
easystr = ' '.join(easytext)

tessdata.to_csv('exp/tessdata.csv')

with open('exp/tessstr.txt', 'w') as f:
    f.write(tessstr)

with open('exp/easystr.txt', 'w') as f:
    f.write(easystr)

