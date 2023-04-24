import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import logging as log
import helpers
import cv2
from config import PATHS

def plot():
    # compare to MOS of dataset, somehow
    data = pd.read_csv('results/cer_ezocr.csv')
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper'
    # print(f'data:\n{data}')
    for num in data.img_num.unique():
        print(f'Plotting normal values for img {num}')
        tmp = data[(data.img_num == num) & (data.comp == 1)]
        ax = tmp.plot.scatter(x='mos',
                              y='cer_comp',
                              c='qual',
                              colormap=colormap,
                              marker=markers[0],
                              label=f'{comps[0]}')
        for idx, comp in enumerate(data.comp.unique()[1:], start=1):
            # print(f'Plotting for comp {comp}')
            tmp = data[(data.img_num == num) & (data.comp == comp)]
            # print(f'data for comp {comp}:\n{tmp}')
            tmp.plot.scatter(x='mos',
                             y='cer_comp',
                             c='qual',
                             colormap=colormap,
                             marker=markers[idx],
                             ax=ax,
                             colorbar=False,
                             label=f'{comps[idx]}')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title(f'Image: {num}')
        plt.xlabel("$MOS$")
        plt.ylabel("$1-CER$")
        plt.savefig(f'images/analyze/mos_ter_ezocr_img{num}.pdf')
        plt.close()

def plot_fitted():
    # compare to MOS of dataset, somehow
    data = pd.read_csv('results/cer_ezocr.csv')
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper'
    for num in data.img_num.unique():
        print(f'Plotting fitted values for img {num}')
        tmp = data.loc[(data.img_num == num) & (data.comp == 1)].copy()
        _, _, fitted_tmp = helpers.nonlinearfitting(tmp['cer_comp'],
                                                    tmp['mos'],
                                                    max_nfev=10000)
        tmp['cer_comp_fitted'] = fitted_tmp
        ax = tmp.plot.scatter(x='mos',
                              y='cer_comp_fitted',
                              c='qual',
                              colormap=colormap,
                              marker=markers[0],
                              label=f'{comps[0]}')
        for idx, comp in enumerate(data.comp.unique()[1:], start=1):
            # print(f'Plotting for comp {comp}')
            tmp = data.loc[(data.img_num == num) & (data.comp == comp)].copy()
            _, _, fitted_tmp = helpers.nonlinearfitting(tmp['cer_comp'],
                                                        tmp['mos'],
                                                        max_nfev=10000)
            tmp['cer_comp_fitted'] = fitted_tmp
            # print(f'data for comp {comp}:\n{tmp}')
            tmp.plot.scatter(x='mos',
                             y='cer_comp_fitted',
                             c='qual',
                             colormap=colormap,
                             marker=markers[idx],
                             ax=ax,
                             colorbar=False,
                             label=f'{comps[idx]}')
        plt.xlim(0, 100)
        plt.ylim(0, 100)
        plt.title(f'Image: {num}')
        plt.xlabel("$MOS$")
        plt.ylabel("$1-CER$")
        plt.savefig(f'images/analyze/mos_ter_fit_ezocr_img{num}.pdf')
        plt.close()

if __name__ == '__main__':
    plot()
    plot_fitted()
