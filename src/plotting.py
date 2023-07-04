import pandas as pd
import matplotlib.pyplot as plt
import helpers
from config import CONFIG, PATHS
import cv2
from itertools import islice
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    })

MARKER_SIZE = 400
FIGSIZE = 10


############################################################################################################
# Distortion plots
############################################################################################################


def plot_cer_dist_quality():
    """
    CER_comp(against gt), seperate by dist type, highlight distortion quality, y(cer) x(qual)
    Single Line plot
    """

    plt.rcParams.update({
        "font.size": 12
        })

    # load dataframe
    data = pd.read_csv(PATHS['results_dist']())


    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            # fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))

            grp_dist = group_ocr.groupby('dist_name')
            for name_dist, group_dist in grp_dist:


                plt.plot(group_dist.groupby('qual')['cer_comp'].mean(),
                         label=name_dist,
                         marker='s')

            plt.xlabel("Distortion quality")
            plt.ylabel("$CER_{comp}$")
            plt.xticks(list(range(1, 6)))
            plt.ylim(0, 100)
            plt.grid()
            plt.legend()
            plt.tight_layout()
            savepath_pdf=PATHS['analyze'](f'cer_dist_quality_{name_target}_{name_ocr}.pdf')
            plt.savefig(savepath_pdf)
            # savepath_png=PATHS['analyze'](f'cer_dist_quality_{name_target}_{name_ocr}.png')
            # plt.savefig(savepath_png)
            plt.title(f"Comparison of CER with target {name_target} for different distortion types with {name_ocr} OCR")
            # plt.show()
            plt.clf()
            plt.close()

    print(f"plotted CER_comp against distortion quality for each distortion type")


def plot_cer_mos_mean():
    """
    Plot non-fitted CER_comp over MOS for different distortion types, mean over all images
    Plot for every distortion type
    """

    # triple font size to make it readable when 3x3 subplots in latex
    plt.rcParams.update({
        "font.size": 36
        })

    data = pd.read_csv(PATHS['results_dist']())
    
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    crit = 'cer_comp'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            grp_dist = group_ocr.groupby('dist') 
            for idx, (name_dist, group_dist) in enumerate(grp_dist):

                fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))

                mean_dist = group_dist.groupby('qual')[[crit, 'mos']].mean().reset_index()
                dist_name = group_dist['dist_name'].iloc[0]

                # plot mean cer/mos over images for each quality
                plt.scatter(mean_dist['mos'],
                                mean_dist[crit],
                                label=dist_name,
                                marker="o",
                                cmap=colormap,
                                c=mean_dist['qual'],
                                s=MARKER_SIZE)

                plt.xlabel("$MOS$")
                plt.ylabel("$CER_{comp}$")
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.grid()

                # make plot square and add colorbar
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax, label='Quality')

                savepath_pdf=PATHS['analyze'](f'mos_cer_{name_target}_mean_{name_ocr}_{dist_name}.pdf')
                plt.savefig(savepath_pdf, bbox_inches='tight', pad_inches=0)
                # savepath_png=PATHS['analyze'](f'mos_cer_{name_target}_mean_{name_ocr}_{dist_name}.png')
                # plt.savefig(savepath_png, bbox_inches='tight', pad_inches=0)

                plt.title(f"Comparison of CER with target {name_target} for {dist_name} with {name_ocr} OCR")
                # plt.show()
                plt.clf()
                plt.close()

    print(f"plotted CER_comp against MOS for each distortion type, mean over all images")


def plot_cer_mos_fitted_mean():
    """
    Plot CER_comp against different targets fitted on all images (total), for different distortion types
    Plot for every distortion type
    """

    plt.rcParams.update({
        "font.size": 36
        })

    data = pd.read_pickle(PATHS['results_dist'](ext='pkl'))
    
    # markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    crit = 'mos_comp_fitted_total'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            grp_dist = group_ocr.groupby('dist') 
            for idx, (name_dist, group_dist) in enumerate(grp_dist):

                fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))

                mean_dist = group_dist.groupby('qual')[[crit, 'cer_comp']].mean().reset_index()
                dist_name = group_dist['dist_name'].iloc[0]
                help = np.arange(0, 100, 0.1)
                model_params = group_dist['model_params_comp_total'].iloc[0]

                # plot point cloud
                plt.scatter(group_dist["mos"],
                            group_dist['cer_comp'],
                            label='single datapoints',
                            marker='.',
                            cmap=colormap,
                            c=group_dist['qual'])
                # plot fitted function
                plt.plot(helpers.model(help, *model_params), help, color='black', linestyle='--', label='fitted model')

                # plot mean cer/mos over images for each quality
                plt.scatter(mean_dist[crit],
                                mean_dist['cer_comp'],
                                label="fitted mean",
                                marker='x',
                                cmap=colormap,
                                c=mean_dist['qual'],
                                s=MARKER_SIZE)

                plt.xlabel("$MOS$")
                plt.ylabel("$CER_{comp}$")
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                plt.grid()
                plt.legend()

                # make plot square and add colorbar
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax, label='Quality')

                savepath_pdf=PATHS['analyze'](f'mos_cer_{name_target}_fitted_mean_{name_ocr}_{dist_name}.pdf')
                plt.savefig(savepath_pdf, bbox_inches='tight', pad_inches=0)
                # savepath_png=PATHS['analyze'](f'mos_cer_{name_target}_fitted_mean_{name_ocr}_{dist_name}.png')
                # plt.savefig(savepath_png, bbox_inches='tight', pad_inches=0)

                plt.title(f"Comparison of CER with target {name_target} for {dist_name} with {name_ocr} OCR (fitted)")
                # plt.show()
                plt.clf()
                plt.close()

    print(f"plotted CER_comp against MOS for each distortion type, mean over all images (fitted)")


############################################################################################################
# codec plots
############################################################################################################

def plot_codec_cer_size():

    plt.rcParams.update({
        "font.size": 12
        })

    # load dataframe
    data = pd.read_csv(PATHS['results_codecs'])

    divider = 1_000_000

    grp_codec_config = data.groupby('codec_config')
    for name_codec_config, group_codec_config in grp_codec_config:

        grp_ocr = group_codec_config.groupby('ocr_algo')

        for name_ocr, group_ocr in grp_ocr:

            # fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))

            data_true = group_ocr.loc[group_ocr.target == "gt"]
            data_pseudo = group_ocr.loc[group_ocr.target == "ref"]

            plt.plot(data_true.loc[(data_true.codec == "vtm")].groupby('q')["size"].mean()/divider,
                        data_true.loc[(data_true.codec == "vtm")].groupby('q')["cer_comp"].mean(),
                        label='VTM true GT',
                        marker='s',
                        color='lightblue')
            plt.plot(data_true.loc[(data_true.codec == "hm")].groupby('q')["size"].mean()/divider,
                        data_true.loc[(data_true.codec == "hm")].groupby('q')["cer_comp"].mean(),
                        label='HM true GT',
                        marker='s',
                        color='blue')
            plt.plot(data_pseudo.loc[(data_pseudo.codec == "vtm")].groupby('q')["size"].mean()/divider,
                        data_pseudo.loc[(data_pseudo.codec == "vtm")].groupby('q')["cer_comp"].mean(),
                        label='VTM pseudo GT',
                        marker='8',
                        color='lightcoral')
            plt.plot(data_pseudo.loc[(data_pseudo.codec == "hm")].groupby('q')["size"].mean()/divider,
                        data_pseudo.loc[(data_pseudo.codec == "hm")].groupby('q')["cer_comp"].mean(),
                        label='HM pseudo GT',
                        marker='8',
                        color='red')

            qvalues = data_true.q.unique()
            xvalues = data_true.loc[(data_true.codec == "vtm")].groupby('q')["size"].mean()/divider
            yvalues = data_true.loc[(data_true.codec == "vtm")].groupby('q')["cer_comp"].mean()

            for q, x, y in zip(qvalues, xvalues, yvalues):
                plt.annotate(f"QP={q}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

            # plt.xlim(0, 0.4)
            plt.ylim(0, 100)
            plt.xlabel("Size in Mbit")
            plt.ylabel("$CER_{comp}$")
            plt.grid()
            plt.legend()
            plt.tight_layout()
            savepath_pdf=PATHS['analyze'](f'codec_cer_size_{name_ocr}_{name_codec_config}.pdf')
            plt.savefig(savepath_pdf)
            # savepath_png=PATHS['analyze'](f'codec_cer_size_{name_ocr}_{name_codec_config}.png')
            # plt.savefig(savepath_png)
            plt.title(f"Comparison of codecs for {name_ocr} with {name_codec_config} codec config")
            # plt.show()
            plt.clf()
            plt.close()

    print(f"plotted CER_comp against size for each codec config and OCR")


def plot_codec_comparison_psnr():
    """
    Plot PSNR curves for different codecs
    """

    # codec_config = CONFIG['codecs_config']
    codec_config = 'default'
    codecs_configs = ['scc', 'default']

    # load dataframe
    data = pd.read_csv(PATHS['results_codecs'])
    # print(data.to_string())

    divider = 1_000_000

    # plot in same figure
    # just select ezocr but doesn't matter in this case

    for codec_config in codecs_configs:
        data_spec = data.loc[(data.ocr_algo == "ezocr") & (data.codec_config == codec_config)]

        # fig = plt.figure(figsize=(FIGSIZE, FIGSIZE))

        plt.plot(data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["psnr"].mean(),
                 label='VTM PSNR',
                 marker='s')
        plt.plot(data_spec.loc[(data_spec.codec == "hm")].groupby('q')["size"].mean()/divider,
                 data_spec.loc[(data_spec.codec == "hm")].groupby('q')["psnr"].mean(),
                 label='HM PSNR',
                 marker='8')

        qvalues = data_spec.q.unique()
        xvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider
        yvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["psnr"].mean()

        for q, x, y in zip(qvalues, xvalues, yvalues):
            plt.annotate(f"QP={q}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')

        # plt.xlim(0, 0.4)
        # plt.ylim(0, 0.36)
        plt.xlabel("Avg size of images in Mbit")
        plt.ylabel("Avg PSNR in dB")
        plt.grid()
        plt.legend()
        plt.tight_layout()
        savepath_pdf=PATHS['analyze'](f'codec_comparison_PSNR_{codec_config}.pdf')
        savepath_png=PATHS['analyze'](f'codec_comparison_PSNR_{codec_config}.png')
        plt.savefig(savepath_pdf)
        plt.savefig(savepath_png)
        plt.title(f"Comparison of codecs PSNR with {codec_config} codec config")
        plt.clf()
        plt.close()
        # plt.show()

    print(f"plotted PSNR against quality for each codec")


############################################################################################################
# other plots
############################################################################################################


def image_diff():
    """
    Plot absolute difference of pixels in images
    """
    # calculate difference between reference and q=50 compressed image
    # load images
    ids = CONFIG['codecs_img_ids_combined']

    ref_paths = helpers.create_paths(PATHS["images_scid_ref"], ids)
    codec_paths = helpers.create_paths(PATHS["images_hm_scc"], ids, [50])

    for ref, codec, id in zip(ref_paths, codec_paths, ids):

        ref_img = cv2.imread(ref, cv2.IMREAD_GRAYSCALE)
        codec_img = cv2.imread(codec, cv2.IMREAD_GRAYSCALE)

        diff = cv2.absdiff(ref_img, codec_img)
        cv2.imwrite(f"images/diff_{id}_q50.png", diff)

    print("calculated difference between reference and q=50 compressed image")


def plot_fit_example():

    plt.rcParams.update({
        "font.size": 12
        })


    subj = [30, 50, 80, 90]*10
    obj = [20, 30, 60, 80]*10

    subj += np.random.normal(-5, 5, len(subj))
    obj += np.random.normal(-5, 5, len(obj))

    beta0 = [np.max(subj), np.min(subj),
             np.mean(obj), np.std(obj)/4]
    MAXFEV = 0
    
    # fitting a curve using the data
    params, _ = curve_fit(helpers.model, obj, subj, method='lm', p0=beta0,
                            maxfev=MAXFEV, ftol=1.5e-08, xtol=1.5e-08)

    t = np.arange(0, 100, 0.001)
    curve = helpers.model(t, *params)
    curve_init = helpers.model(t, *beta0)
    subj_fit = helpers.model(np.array(obj), *params)
    p_r = scipy.stats.pearsonr(subj, obj)[0]
    p_s = scipy.stats.spearmanr(subj, obj)[0]
    p_r_fit = scipy.stats.pearsonr(subj_fit, obj)[0]
    p_s_fit = scipy.stats.spearmanr(subj_fit, obj)[0]

    plt.plot(subj, obj, 'o', label='data')
    plt.plot(subj_fit, obj, 'o', label='fitted')
    plt.plot(curve_init, t, label='model$_{init}$')
    plt.plot(curve, t, label='model$_{fitted}$')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.ylabel("objective value")
    plt.xlabel("subjective value")
    tex_p_r = "$r_p=$"
    tex_p_r_fit = "$r_p^{fit}=$"
    tex_p_s = f"$r_s=$"
    tex_p_s_fit = "$r_s^{fit}=$"
    text = f"{tex_p_r}{p_r:.2f}\n{tex_p_r_fit}{p_r_fit:.2f}\n{tex_p_s}{p_s:.2f}\n{tex_p_s_fit}{p_s_fit:.2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey')

    plt.text(0.02, 0.7, text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=props)

    # draw lines from data points to fitted points
    for i in range(len(subj)):
        plt.plot([subj[i], subj_fit[i]], [obj[i], obj[i]], 'k--', alpha=0.2)

    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("exp/fit_example.pdf")
    plt.savefig("exp/fit_example.png")
    # plt.show()
    plt.close()

    print("plotted fitting example")

def pipeline():
    """
    Run all functions to plot everything in one go
    """

    # basic cer over distortion quality
    plot_cer_dist_quality()

    # subplots for different distortions (single images)
    # plot_cer_mos_sub()
    # plot_cer_fitted_mos_sub()

    # subplots for different distortions (mean over all images)
    plot_cer_mos_mean()
    plot_cer_mos_fitted_mean()

    # plot codec comparison
    plot_codec_cer_size()

    # plot psnr of different codecs for different qps
    # plot_codec_comparison_psnr()

    # absolute difference of pixels in images
    # image_diff()

    # example for fitting
    plot_fit_example()


if __name__ == '__main__':

    pass
    # pipeline()
    # plot_cer_dist_quality()
    plot_fit_example()
