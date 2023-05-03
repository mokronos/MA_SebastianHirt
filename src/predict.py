import helpers
import logging as log

log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
# log.disable(level=log.DEBUG)

# maybe put paths in config json file
ref_dir = 'data/raw/scid/ReferenceSCIs'
dist_dir = 'data/raw/scid/DistortedSCIs'

image_ext = '.bmp'
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

# generate list with paths
paths = []
names = []
for num in numbers:
    for comp in compressions:
        for qual in qualities:
            img_name = f'{prefix}{num}_{comp}_{qual}'
            img_path = f'{dist_dir}/{img_name}{image_ext}'
            paths.append(img_path)
            names.append(img_name)

log.debug(f'created list of paths: {paths}')
algos = ['tess', 'ezocr']
algos = ['ezocr']
algos = ['tess']

save_dir = 'results/pred'

# run prediction
for algo in algos:
    preds = helpers.pred_data(paths, algo=algo)

    # save predictions
    for pred, name in zip(preds, names):
        log.debug(f'saving prediction for {name} with {algo} ...')
        pred_path = f'{save_dir}/{algo}/comp/{name}'
        pred.to_csv(f'{pred_path}.csv', index=False)
        text = helpers.csv_to_text(pred)
        with open(f'{pred_path}.txt', 'w') as f:
            f.write(text)

log.info('done')
