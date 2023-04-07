# path stuff
import logging as log
import pytesseract
import easyocr
from PIL import Image
import pandas as pd
import cv2

log.basicConfig(level=log.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
# log.disable(log.CRITICAL)

# image path for sample image
# log.debug(f"loaded image from {image_path}")

# tess_result = pytesseract.image_to_data(Image.open(image_path), lang='eng', output_type=pytesseract.Output.DATAFRAME)

# easy_result = easyocr.Reader(['en']).readtext(image_path, width_ths=0.1)
# # convert list of tuples to dataframe, format: ([[x1, y1], [x2, y2], [x3, y3], [x4, y4]], text, conf)

# log.debug(f"easyocr result: {easy_result[0]}")
# df = pd.DataFrame()
# left, top, right, bottom, text, conf = [], [], [], [], [], []
# for line in easy_result:
#     left.append(line[0][0][0])
#     top.append(line[0][0][1])
#     right.append(line[0][2][0])
#     bottom.append(line[0][2][1])
#     text.append(line[1])
#     conf.append(line[2])
    
# df[['left', 'top', 'right', 'bottom', 'text', 'conf']] = pd.DataFrame([left, top, right, bottom, text, conf]).T

# tess_result = tess_result[~tess_result['text'].isna()]
# sorted_tess = tess_result.sort_values(['top', 'left']).reset_index(drop=True)
# sorted_tess['bottom'] = sorted_tess['top'] + sorted_tess['height']
# sorted_tess['right'] = sorted_tess['left'] + sorted_tess['width']
# sorted_easy = df.sort_values(['top', 'left']).reset_index(drop=True)


# log.debug(f"easyocr result: {sorted_easy}")
# log.debug(f"tesseract result: {sorted_tess}")

# # draw bounding boxes around text in image
# def add_bounding_boxes(image_path, data, method='easyocr'):
#     img = cv2.imread(image_path)
#     for idx, line in data.iterrows():
#         x1, y1, x2, y2 = int(line['left']), int(line['top']), int(line['right']), int(line['bottom'])
#         img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
#         cv2.putText(img, str(idx), (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

#     cv2.imwrite(f'exp/{method}_result.jpg', img)

# add_bounding_boxes(image_path, sorted_easy, method='easyocr')
# add_bounding_boxes(image_path, sorted_tess, method='tesseract')

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

tessdata = tessdata[~tessdata['text'].isna()]

tesstext = tessdata['text'].to_list()
tessstr = ' '.join(tesstext)

log.debug(f"tesseract result: {tessdata['text'].to_list()}")
with open('exp/tessdata.txt', 'w') as f:
    f.write(tessstr)

with open('exp/tessstr.txt', 'w') as f:
    f.write(tessstr)

