# load data
# define paths
import helpers
import pandas as pd
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


# compare to MOS of dataset, somehow
data = pd.read_csv('results/ter.csv')
data.dropna(inplace=True)
print(data[['ter_comp', 'mos']].corr())

prcc, scrcc, adj_mos = helpers.nonlinearfitting(list(data['ter']), list(data['mos']), max_nfev=3000)
print('Pearson correlation coefficient: ', prcc)
print('Spearman correlation coefficient: ', scrcc)
print('Adjusted MOS: ', adj_mos)
data['adj_mos'] = adj_mos

data.plot.scatter(x='adj_mos', y='ter')
# plt.xlim(0, 100)
# plt.ylim(0, 100)
plt.grid()
plt.show()
