# load data
# define paths
import helpers
import numpy as np
import matplotlib.pyplot as plt


# maybe put paths in config json file
ref_dir = 'data/raw/scid/ReferenceSCIs'
dist_dir = 'data/raw/scid/DistortedSCIs'
mos_path = 'data/raw/scid/MOS_SCID.txt'
gt_dir = 'data/gt/scid'

image_ext = '.bmp'
label_ext = '.txt'
prefix = 'SCI'

# get filenames
# numbers = list(range(1, 41))
# selected subset of images with a lot of text focus and cleaned ground truth
amt = 3
numbers = [1, 3, 4, 5, 29]
numbers = numbers[:amt]
numbers = [x if x > 9 else f'0{x}' for x in numbers]
compressions = [1, 2, 3, 4, 5, 6, 7, 8, 9]
compressions = compressions[-amt:]
qualities = [1, 2, 3, 4, 5]
qualities = qualities[-amt:]
algos = ['tess', 'ezocr']

# read mos data into dataframe
data = helpers.read_mos(mos_path)
print(data.head())

for algo in algos:
    data[f'ter_{algo}'] = np.nan
    for num in numbers:
        for comp in compressions:
            for qual in qualities:
                # just text, ignore bounding box/position
                # run on different compression levels (SCID dataset)
                img_name = f'{prefix}{num}_{comp}_{qual}'
                print(f'Running for {img_name} with {algo} ...')
                label_name = f'{prefix}{num}'
                img_path = f'{dist_dir}/{img_name}{image_ext}'
                label_path = f'{gt_dir}/{label_name}_gt{label_ext}'
                pred_path = f'results/pred/{algo}/comp/{img_name}_pred{label_ext}'

                pred = helpers.pred_img(img_path, label_path, algo=algo)
                print(f'prediction with {algo}:')

                # save prediction
                with open(pred_path, 'w') as f:
                    for line in pred:
                        f.write(f"{line}\n")

                # load label, compare and save text-error-rate
                with open(label_path) as f:
                    label = f.read()
                ter = helpers.text_error_rate('\n'.join(pred), label)

                data.loc[(data.img_num == int(num))
                         & (data.comp == comp)
                         & (data.qual == qual), f'ter_{algo}'] = ter

    # compare to MOS of dataset, somehow
    data[f'ter_comp_{algo}'] = (1 - data[f'ter_{algo}']) * 100

# dont overwrite ter.csv
data.to_csv('results/ter2.csv', index=False)
data.dropna(inplace=True)

# plot
# data.plot.scatter(x='mos', y='ter_comp')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.grid()
# plt.show()
