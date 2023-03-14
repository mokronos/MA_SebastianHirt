# load data
# define paths
import helpers
import numpy as np
import matplotlib.pyplot as plt


# split functions into i/o and processing


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

data = helpers.read_mos(mos_path)
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

            ter = helpers.pred_img(img_path, label_path)
            data.loc[(data.img_num == int(num))
                     & (data.comp == comp)
                     & (data.qual == qual), 'ter'] = ter

# compare to MOS of dataset, somehow
data['ter_comp'] = (1 - data['ter']) * 100
# dont overwrite ter.csv
data.to_csv('results/ter2.csv', index=False)
data.dropna(inplace=True)

# plot
# data.plot.scatter(x='mos', y='ter_comp')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
# plt.grid()
# plt.show()
