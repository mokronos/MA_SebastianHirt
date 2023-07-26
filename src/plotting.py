import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import gridspec
import helpers
from config import CONFIG, PATHS
import cv2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import pytesseract
from PIL import Image

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
            plt.ylabel("CER$_{\mathrm{c}}$")
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
                plt.scatter(mean_dist[crit],
                            mean_dist['mos'],
                            label=dist_name,
                            marker="o",
                            cmap=colormap,
                            c=mean_dist['qual'],
                            s=MARKER_SIZE)

                plt.xlabel("$\overline{\mathrm{CER}}_{\mathrm{c}}$")
                plt.ylabel("$\overline{\mathrm{MOS}}$")
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                ticks = [0, 20, 40, 60, 80, 100]
                plt.xticks(ticks)
                plt.yticks(ticks)
                plt.grid()

                # make plot square and add colorbar
                ax = plt.gca()
                ax.set_aspect('equal', adjustable='box')
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(cax=cax, label='Quality')

                savepath_pdf=PATHS['analyze'](f'mos_cer_{name_target}_mean_{name_ocr}_{dist_name}.pdf')
                # pad_inches to leave overline there
                plt.savefig(savepath_pdf, bbox_inches='tight', pad_inches=0.1)
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

                mean_dist = group_dist.groupby('qual')[['mos',crit]].mean().reset_index()
                dist_name = group_dist['dist_name'].iloc[0]
                help = np.arange(0, 100, 0.001)
                model_params = group_dist['model_params_comp_total'].iloc[0]

                # plot point cloud
                plt.scatter(group_dist['cer_comp'],
                            group_dist["mos"],
                            label='Data',
                            marker='.',
                            cmap=colormap,
                            c=group_dist['qual'])

                # plot point cloud for fitted values
                plt.scatter(group_dist[crit],
                            group_dist["mos"],
                            label='Data$_{\mathrm{fit}}$',
                            marker='s',
                            cmap=colormap,
                            c=group_dist['qual'])

                # plot point cloud for fitted values (turned)
                # plt.scatter(group_dist['cer_comp'],
                #             group_dist[crit],
                #             label='Single datapoints (fitted)(turned)',
                #             marker='*',
                #             cmap=colormap,
                #             c=group_dist['qual'])

                # plot diagonal line
                # plt.plot(help, help, color='red', linestyle='-', label='Diagonal')

                # plot fitted function
                plt.plot(help,
                         helpers.model(help, *model_params),
                         color='black',
                         linestyle='--',
                         label='Model$_{\mathrm{fit}}$')

                # plot mean cer/mos over images for each quality
                plt.scatter(mean_dist[crit],
                            mean_dist['mos'],
                            label="Mean$_{\mathrm{fit}}$",
                            marker='x',
                            cmap=colormap,
                            c=mean_dist['qual'],
                            s=MARKER_SIZE)

                plt.xlabel("CER$_{\mathrm{c}}$/MOS$_{\mathrm{p}}$")
                plt.ylabel("MOS")
                plt.xlim(0, 100)
                plt.ylim(0, 100)
                ticks = [0, 20, 40, 60, 80, 100]
                plt.xticks(ticks)
                plt.yticks(ticks)
                plt.grid()

                # set color of legend markers to black
                leg = plt.legend(loc='upper left', borderpad=0.2, handletextpad=0.2, labelspacing=0.2)
                for lh in leg.legend_handles:
                    lh.set_color('black')
                    lh.set_alpha(1)

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
            plt.ylabel("$\overline{\mathrm{CER}}_{\mathrm{c}}$")
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
    id = 4

    ref_path = PATHS["images_scid_ref"](id)
    codec_vtm_default = PATHS["images_codec"](id, q=50, codec_config='default', codec='vtm')
    codec_hm_default = PATHS["images_codec"](id, q=50, codec_config='default', codec='hm')
    codec_vtm_scc = PATHS["images_codec"](id, q=50, codec_config='scc', codec='vtm')
    codec_hm_scc = PATHS["images_codec"](id, q=50, codec_config='scc', codec='hm')



    ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
    codec_vtm_default_img = cv2.imread(codec_vtm_default, cv2.IMREAD_GRAYSCALE)
    codec_hm_default_img = cv2.imread(codec_hm_default, cv2.IMREAD_GRAYSCALE)
    codec_vtm_scc_img = cv2.imread(codec_vtm_scc, cv2.IMREAD_GRAYSCALE)
    codec_hm_scc_img = cv2.imread(codec_hm_scc, cv2.IMREAD_GRAYSCALE)

    codec_vtm_default_diff = cv2.absdiff(ref_img, codec_vtm_default_img)
    codec_hm_default_diff = cv2.absdiff(ref_img, codec_hm_default_img)
    codec_vtm_scc_diff = cv2.absdiff(ref_img, codec_vtm_scc_img)
    codec_hm_scc_diff = cv2.absdiff(ref_img, codec_hm_scc_img)

    cv2.imwrite(f"images/codec_vtm_default_diff_50_SCI{id}.png", codec_vtm_default_diff)
    cv2.imwrite(f"images/codec_hm_default_diff_50_SCI{id}.png", codec_hm_default_diff)
    cv2.imwrite(f"images/codec_vtm_scc_diff_50_SCI{id}.png", codec_vtm_scc_diff)
    cv2.imwrite(f"images/codec_hm_scc_diff_50_SCI{id}.png", codec_hm_scc_diff)

    print("calculated difference between reference and q=50 compressed image")


def plot_fit_example():

    plt.rcParams.update({
        "font.size": 12
        })

    # fix random seed for reproducibility
    np.random.seed(7)

    subj = [30, 50, 80, 80]*10
    obj = [20, 30, 60, 80]*10

    subj += np.random.normal(-5, 5, len(subj))
    obj += np.random.normal(-5, 5, len(obj))

    # beta0 = [np.max(subj), np.min(subj),
               #          np.mean(obj), np.std(obj)/4,
               #          0]
    # beta0 = [-100,1,1,0.5,50]
    beta0 = [np.max(subj), np.min(subj),
             np.mean(obj), 1]
    MAXFEV = 0

    # fitting a curve using the data
    params, _ = curve_fit(helpers.model, obj, subj, method='lm', p0=beta0,
                          maxfev=MAXFEV, ftol=1.5e-08, xtol=1.5e-08)

    t = np.arange(0, 100, 0.001)
    curve = helpers.model(t, *params)
    curve_init = helpers.model(t, *beta0)
    subj_fit = helpers.model(np.array(obj), *params)
    p_r = scipy.stats.pearsonr(subj, obj)[0]
    p_r_fit = scipy.stats.pearsonr(subj, subj_fit)[0]
    p_s = scipy.stats.spearmanr(subj, obj)[0]
    p_s_fit = scipy.stats.spearmanr(subj_fit, subj)[0]

    plt.plot(subj_fit, subj, 's', label='Data$_{\mathrm{fit}}$')
    plt.plot(obj, subj, '.', label='Data')
    plt.plot(t, curve_init, label='Model$_{\mathrm{ini}}$', color='green', linestyle='--')
    plt.plot(t, curve, label='Model$_{\mathrm{fit}}$', color='black', linestyle='--')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.ylabel("MOS")
    plt.xlabel("CER$_{\mathrm{c}}$/MOS$_{\mathrm{p}}$")
    tex_p_r = "$r_p=$"
    tex_p_r_fit = "$r_p^{fit}=$"
    tex_p_s = f"$r_s=$"
    tex_p_s_fit = "$r_s^{fit}=$"
    text = f"{tex_p_r}{p_r:.2f}\n{tex_p_r_fit}{p_r_fit:.2f}\n{tex_p_s}{p_s:.2f}\n{tex_p_s_fit}{p_s_fit:.2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5, edgecolor='grey')

    # plt.text(0.02, 0.7, text, transform=plt.gca().transAxes,
               #         verticalalignment='top', bbox=props)

    # draw lines from data points to fitted points
    for x, y, y_fit in zip(obj, subj, subj_fit):
        plt.plot([x, y_fit], [y, y], color='grey', linestyle='--', linewidth=0.5)
        

    ax = plt.gca()
    ax.set_aspect('equal', adjustable='box')

    plt.legend()
    plt.tight_layout()
    plt.grid()
    plt.savefig("exp/fit_example.pdf")
    plt.savefig("exp/fit_example.png")
    # plt.show()
    plt.close()

    print("plotted fitting example")

def plot_bjontegaard_example():

    rateA = [0.1, 0.2, 0.3, 0.4]
    rateB = [0.08, 0.16, 0.24, 0.32]

    distA = [20, 33, 40, 43]
    distB = [25, 38, 45, 48]


    plt.plot(rateB, distB, label="Codec B", marker='o')
    plt.plot(rateA, distA, label="Codec A", marker='o')
    plt.xlabel("Bitrate in Mbit/s")
    plt.ylabel("CER$_{\mathrm{c}}$")

    plt.legend()
    plt.grid()

    color = "yellow"
    # color the area between the two curves
    plt.fill_between(rateA, distA, distB, color=color)
    plt.fill_between(rateB, distA, distB, color=color)

    # make areas above and below curves white
    plt.fill_between([rateB[0], rateA[-1]], [distA[-1], distA[-1]], max(distB), color='white')
    plt.fill_between([rateB[0], rateA[-1]], [distB[0], distB[0]], min(distA), color='white')

    plt.savefig("exp/bjontegaard_example.pdf")
    plt.savefig("exp/bjontegaard_example.png")

    # plt.show()

    print("plotted bjontegaard example")

def bbox_order(id=1, algo="ezocr", sort=True):
    """
    check how to order tesseract results that the text is in lines
    """

    save_paths_csv = helpers.create_paths(PATHS["pred_ref"],
                                            [id],
                                            algo=algo, ext="csv")

    save_path = save_paths_csv[0]
    print(save_path)

    pred_csv = pd.read_csv(save_path)
    print(pred_csv)

    # load image
    img_path = PATHS["images_scid_ref"](num=id)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # clean data
    pred_csv = pred_csv.loc[~pred_csv["text"].isna()]
    pred_csv = pred_csv.loc[pred_csv["text"].str.strip() != ""]

    if sort:
        new_data = helpers.sort_boxes(pred_csv)
    else:
        if algo == "ezocr":
            new_data = pred_csv
        else:
            # predict with tesseract
            with Image.open(img_path) as image:
                pred = pytesseract.image_to_data(image, output_type=pytesseract.Output.DATAFRAME, config="--oem 1")
                pred = pred.loc[~pred["text"].isna()]
                pred = pred.loc[pred["text"].str.strip() != ""]
                pred['right'] = pred['left'] + pred['width']
                pred['bottom'] = pred['top'] + pred['height']
                new_data = pred
                new_data.reset_index(inplace=True, drop=True)


    plt.imshow(img)

    # draw text boxes
    for i, row in new_data.iterrows():
        x, y, w, h = row["left"], row["bottom"], row["right"]-row["left"], row["top"]-row["bottom"]
        text = f"({i})"
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, fill=False, color="green", linewidth=0.5))
        # put text under bounding box
        plt.text(x, y+12, text, color="green", fontsize=4, weight="bold")

    plt.axis('off')
    plt.savefig(f"images/bbox_order_{algo}{'_sorted' if sort else ''}.pdf", dpi=150, bbox_inches='tight', pad_inches=0)
    plt.close()
    plt.clf()

def dataset_analysis():

    plt.rcParams.update({
        "font.size": 24
        })

    data = pd.read_csv(PATHS['results_dist']())
    ids = CONFIG["scid_img_ids"]

    # filter data
    data = data.loc[data["img_num"].isin(ids)]

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:


            # plot cer against mos for all selected ids
            plt.plot(group_ocr["cer_comp"], group_ocr["mos"], marker='.', linestyle='None')
            plt.xlabel("CER$_{\mathrm{c}}$")
            plt.ylabel("MOS")
            plt.grid()
            plt.xlim(0, 100)
            plt.ylim(0, 100)
            ticks = [0, 20, 40, 60, 80, 100]
            plt.xticks(ticks)
            plt.yticks(ticks)
            ax = plt.gca()
            ax.set_aspect('equal', adjustable='box')
            plt.savefig(f"images/cer_mos_overview_{name_target}_{name_ocr}.pdf", bbox_inches='tight', pad_inches=0)
            plt.close()
            plt.clf()

def dataset_overview():

    ids = list(range(1, 41))
    ref_paths = helpers.create_paths(PATHS["images_scid_ref"], ids)

    save_path = "./images/"

    fig = plt.figure(figsize=(5, 7))
    gs = gridspec.GridSpec(8, 5, wspace=0.02, hspace=0.)

    for idx, g in enumerate(gs):
        if len(ref_paths) == 0:
            break
        ax = plt.subplot(g)
        ax.imshow(plt.imread(ref_paths.pop(0)))
        ax.axis('off')
        ax.set_title(f'SCI {idx+1}', fontsize=6, y=0.88)


    plt.savefig(save_path + "reference_images.pdf", dpi=300, bbox_inches='tight', pad_inches=0)
    # plt.savefig(save_path + "reference_images.png", dpi=300)
    # # 35MB laggy
    # plt.savefig(save_path + "reference_images_HD.pdf", dpi=3000)
    # plt.savefig(save_path + "reference_images_HD.png", dpi=3000)

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
    image_diff()

    # example for fitting
    plot_fit_example()
    plot_bjontegaard_example()

    # bbox order examples
    bbox_order(id=1, algo="tess", sort=True)
    bbox_order(id=1, algo="tess", sort=False)
    bbox_order(id=1, algo="ezocr", sort=False)

    # dataset
    dataset_analysis()
    dataset_overview()


if __name__ == '__main__':

    pass
    # pipeline()
    # plot_fit_example()
    # plot_codec_cer_size()
    # plot_bjontegaard_example()
    # plot_cer_dist_quality()
    # plot_cer_mos_mean()
    # plot_cer_mos_fitted_mean()
    # bbox_order(id=1, algo="tess", sort=True)
    # bbox_order(id=1, algo="tess", sort=False)
    # bbox_order(id=1, algo="ezocr", sort=False)
    # dataset_overview()
    image_diff()



