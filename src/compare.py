import helpers
import logging as log
import numpy as np

log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
# log.disable(level=log.DEBUG)

# maybe put paths in config json file
ref_dir = 'data/raw/scid/ReferenceSCIs'
dist_dir = 'data/raw/scid/DistortedSCIs'
mos_path = 'data/raw/scid/MOS_SCID.txt'
gt_dir = 'data/gt/scid/line'

image_ext = '.bmp'
label_ext = '.txt'
prefix = 'SCI'

# get filenames
# numbers = list(range(1, 41))
# selected subset of images with a lot of text focus and cleaned ground truth
amt = 3
numbers = [1, 3, 4, 5, 29]
# numbers = numbers[:amt]
numbers = [x if x > 9 else f'0{x}' for x in numbers]
compressions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# compressions = compressions[-amt:]
qualities = [1, 2, 3, 4, 5]
# qualities = qualities[-amt:]

# read mos data into dataframe
data = helpers.read_mos(mos_path)

# algo = 'ezocr'
algo = 'tess'

data[f'cer'] = np.nan
for num in numbers:
    for comp in compressions:
        for qual in qualities:
            # just text, ignore bounding box/position
            # run on different compression levels (SCID dataset)
            img_name = f'{prefix}{num}_{comp}_{qual}'
            label_name = f'{prefix}{num}'
            label_path = f'{gt_dir}/{label_name}_gt{label_ext}'
            pred_path = f'results/pred/{algo}/comp/{img_name}{label_ext}'

            log.debug(f'Comparing for {img_name} with {algo} ...')
            pred = helpers.load_line_text(pred_path)

            # load label, compare and save text-error-rate
            with open(label_path) as f:
                label = f.read()
            cer = helpers.char_error_rate(pred, label)

            data.loc[(data.img_num == int(num))
                     & (data.comp == comp)
                     & (data.qual == qual), f'cer'] = cer

# compare to MOS of dataset, somehow
data[f'cer_comp'] = (1 - data[f'cer']) * 100

# dont overwrite cer.csv
data.to_csv(f'results/cer_{algo}.csv', index=False)
