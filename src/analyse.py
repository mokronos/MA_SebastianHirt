import pandas as pd
import matplotlib.pyplot as plt
import logging as log

log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
log.disable(level=log.DEBUG)

# compare to MOS of dataset, somehow
data = pd.read_csv('results/cer_ezocr.csv')
data.dropna(inplace=True)

# plot
markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
colormap = 'copper'
for num in data.img_num.unique():
    log.info(f'Plotting for img {num}')
    tmp = data[(data.img_num == num) & (data.comp == 9)]
    ax = tmp.plot.scatter(x='mos', y='cer_comp', c='qual', colormap=colormap, marker=markers[8], label='comp 9')
    for idx, comp in enumerate(data.comp.unique()[:-1]):
        tmp = data[(data.img_num == num) & (data.comp == comp)]
        # log.info(f'data for comp {comp}:\n{tmp}')
        tmp.plot.scatter(x='mos',
                         y='cer_comp',
                         c='qual',
                         colormap=colormap,
                         marker=markers[idx],
                         ax=ax,
                         colorbar=False,
                         label=f'{comps[comp]}')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.title(f'Image: {num}')
    plt.xlabel("$MOS$")
    plt.ylabel("$CER_{comp}$")
    plt.savefig(f'images/analyze/mos_ter_ezocr_img{num}.pdf')
    plt.close()
