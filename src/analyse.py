# load data
# define paths
import helpers
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# split functions into i/o and processing
ref_dir = 'data/raw/scid/ReferenceSCIs'
dist_dir = 'data/raw/scid/DistortedSCIs'
mos_path = 'data/raw/scid/MOS_SCID.txt'
gt_dir = 'data/gt/scid'

image_ext = '.bmp'
label_ext = '.txt'
prefix = 'SCI'


# compare to MOS of dataset, somehow
data = pd.read_csv('results/ter2.csv')
data.dropna(inplace=True)
# print(data[['ter_comp', 'mos']].corr())

# prcc, scrcc, adj_mos = helpers.nonlinearfitting(list(data['ter']), list(data['mos']), max_nfev=3000)
# print('Pearson correlation coefficient: ', prcc)
# print('Spearman correlation coefficient: ', scrcc)
# print('Adjusted MOS: ', adj_mos)
# data['adj_mos'] = adj_mos

algos = ['tess', 'ezocr']

ax = data.plot.scatter(x='mos', y=f'ter_comp_{algos[0]}', color='blue')
data.plot.scatter(x='mos', y=f'ter_comp_{algos[1]}', ax=ax, color='red')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.grid()
plt.show()
