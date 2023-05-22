import pandas as pd
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import logging as log
import helpers
from config import CONFIG, PATHS

def plot():
    # compare to MOS of dataset, somehow
    data = pd.read_csv(PATHS['results'])
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper_r'
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
    data = pd.read_csv(PATHS['results'])
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper_r'
    for num in data.img_num.unique():
        print(f'Plotting fitted values for img {num}')
        tmp = data.loc[(data.img_num == num) & (data.comp == 1)].copy()
        fitted_tmp = helpers.nonlinearfitting(tmp['cer_comp'],
                                              tmp['mos'],
                                              max_nfev=20000)
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
            fitted_tmp = helpers.nonlinearfitting(tmp['cer_comp'],
                                                  tmp['mos'],
                                                  max_nfev=20000)
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


def plot_sub():
    # compare to MOS of dataset, somehow
    data = pd.read_csv(PATHS['results'])
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper_r'
    # print(f'data:\n{data}')
    for num in data.img_num.unique():
        print(f'Plotting normal values for img {num}')
        # define subplots
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        for idx, (comp,ax) in enumerate(zip(data.comp.unique(), axs.flatten())):
            # print(f'Plotting for comp {comp}')
            tmp = data[(data.img_num == num) & (data.comp == comp)]
            # print(f'data for comp {comp}:\n{tmp}')
            tmp.plot.scatter(x='mos',
                             y='cer_comp',
                             c='qual',
                             colormap=colormap,
                             marker=markers[idx],
                             ax=ax,
                             colorbar=True)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("$MOS$")
            ax.set_ylabel("$1-CER$")
            ax.set_title(f'{comps[idx]} ({comp})')
        fig.suptitle(f'Image: {num}')
        plt.tight_layout()
        plt.savefig(f'images/analyze/mos_ter_ezocr_sub_img{num}.pdf')
        plt.close()

def plot_fitted_sub():
    # compare to MOS of dataset, somehow
    data = pd.read_csv(PATHS['results'])
    data.dropna(inplace=True)

    # plot
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    comps = ["GN", "GB", "MB", "CC", "JPEG", "JPEG2000", "CSC", "HEVC-SCC", "CQD"]
    colormap = 'copper_r'
    # print(f'data:\n{data}')
    for num in data.img_num.unique():
        print(f'Plotting fitted values for img {num}')
        # define subplots
        fig, axs = plt.subplots(3, 3, figsize=(10, 10))

        for idx, (comp,ax) in enumerate(zip(data.comp.unique(), axs.flatten())):
            # print(f'Plotting for comp {comp}')
            tmp = data.loc[(data.img_num == num) & (data.comp == comp)].copy()
            fitted_tmp = helpers.nonlinearfitting(tmp['cer_comp'],
                                                  tmp['mos'],
                                                  max_nfev=20000)
            tmp['cer_comp_fitted'] = fitted_tmp
            # print(f'data for comp {comp}:\n{tmp}')
            tmp.plot.scatter(x='mos',
                             y='cer_comp_fitted',
                             c='qual',
                             colormap=colormap,
                             marker=markers[idx],
                             ax=ax,
                             colorbar=True)
            ax.set_xlim(0, 100)
            ax.set_ylim(0, 100)
            ax.set_xlabel("$MOS$")
            ax.set_ylabel("$1-CER$")
            ax.set_title(f'{comps[idx]} ({comp})')
        fig.suptitle(f'Image: {num}')
        plt.tight_layout()
        plt.savefig(f'images/analyze/mos_ter_fit_ezocr_sub_img{num}.pdf')
        plt.close()

def plot_codec_comparison():

    # load dataframe
    data = pd.read_csv(PATHS['results_codecs'])

    ocr_algos = CONFIG['ocr_algos']

    divider = 1_000_000

    # plot in same figure
    for algo in ocr_algos:
        data_spec = data.loc[(data.ocr_algo == algo)]
        plt.plot(data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_true"].mean(),
                 label='VTM true GT',
                 marker='s')
        plt.plot(data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_pseudo"].mean(),
                 label='VTM pseudo GT',
                 marker='8')
        plt.plot(data_spec.loc[(data_spec.codec == "hm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "hm")].groupby('q')["cer_true"].mean(),
                 label='HM true GT',
                 marker='^')
        plt.plot(data_spec.loc[(data_spec.codec == "hm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "hm")].groupby('q')["cer_pseudo"].mean(),
                 label='HM pseudo GT',
                 marker='v')

        qvalues = data_spec.q.unique()
        xvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider
        yvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_pseudo"].mean()

        for q, x, y in zip(qvalues, xvalues, yvalues):
            plt.annotate(f"QP={q}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

                            
                        

        plt.xlim(0, 0.4)
        plt.ylim(0, 0.35)
        plt.xlabel("Avg size of images in Mbit")
        plt.ylabel("CER")
        plt.title(f"Comparison of codecs for {algo}")
        plt.grid()
        plt.legend()
        savepath=PATHS['analyze'](f'codec_comparison_{algo}.pdf')
        plt.savefig(savepath)
        plt.show()


if __name__ == '__main__':
    # plot()
    # plot_sub()
    # plot_fitted()
    # plot_fitted_sub()
    plot_codec_comparison()
