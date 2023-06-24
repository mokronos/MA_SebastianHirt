import pandas as pd
import matplotlib.pyplot as plt
import helpers
from config import CONFIG, PATHS
import cv2
from itertools import islice

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    # "font.serif": ["Computer Modern Roman"],
    })



############################################################################################################
# Distortion plots
############################################################################################################


def plot_cer_dist_quality():
    """
    CER_comp(against gt), seperate by dist type, highlight distortion quality, y(cer) x(qual)
    Single Line plot
    """

    # load dataframe
    data = pd.read_csv(PATHS['results_dist'])


    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            grp_dist = group_ocr.groupby('dist_name')
            for name_dist, group_dist in grp_dist:

                plt.plot(group_dist.groupby('qual')['cer_comp'].mean(),
                         label=name_dist,
                         marker='s')

            plt.xlabel("Distortion quality")
            plt.ylabel("$\overline{CER}_{comp}$")
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


def plot_cer_mos_sub():
    """
    Plot non fitted CER against MOS for each distortion type, for single image (not mean)
    Subplots
    """

    data = pd.read_csv(PATHS['results_dist'])
    
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:

        grp_ocr = group_target.groupby('ocr_algo')

        for name_ocr, group_ocr in grp_ocr:

            grp_img = group_ocr.groupby('img_num')
            grp_img = islice(grp_img, 1)
            for name_img, group_img in grp_img:

                grp_dist = group_img.groupby('dist') 

                fig, axs = plt.subplots(3, 3, figsize=(10, 10))

                for idx, ((name_dist, group_dist), ax) in enumerate(zip(grp_dist, axs.flatten())):
                    im = ax.scatter(group_dist['mos'],
                                    group_dist['cer_comp'],
                                    label=group_dist['dist_name'].iloc[0],
                                    marker=markers[idx],
                                    cmap=colormap,
                                    c=group_dist['qual'])


                    fig.colorbar(im, ax=ax,label='Quality')
                    ax.set_xlabel("$MOS$")
                    ax.set_ylabel("$CER_{comp}$")
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    ax.set_title(f'Dist. type: {group_dist["dist_name"].iloc[0]}')

                plt.tight_layout()
                plt.savefig(f"images/mos_cer_{name_target}_sub_{name_ocr}_img{name_img}.pdf")
                # plt.show()
                plt.clf()
                plt.close()

    print(f"plotted CER_comp against MOS for each distortion type, for single image (not mean)")


def plot_cer_fitted_mos_sub():
    """
    Plot fitted CER against MOS for each distortion type, for single image (not mean)
    Subplots
    """

    data = pd.read_csv(PATHS['results_dist'])
    
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            grp_img = group_ocr.groupby('img_num')

            # only for one image, comment to plot all
            grp_img = islice(grp_img, 1)
            for name_img, group_img in grp_img:

                grp_dist = group_img.groupby('dist') 

                fig, axs = plt.subplots(3, 3, figsize=(10, 10))

                for idx, ((name_dist, group_dist), ax) in enumerate(zip(grp_dist, axs.flatten())):
                    im = ax.scatter(group_dist['mos'],
                                    group_dist['cer_comp_fitted'],
                                    label=group_dist['dist_name'].iloc[0],
                                    marker=markers[idx],
                                    cmap=colormap,
                                    c=group_dist['qual'])


                    fig.colorbar(im, ax=ax,label='Quality')
                    ax.set_xlabel("$MOS$")
                    ax.set_ylabel("$CER_{comp} (fitted)$")
                    ax.set_xlim(0, 100)
                    ax.set_ylim(0, 100)
                    ax.set_title(f'Dist. type: {group_dist["dist_name"].iloc[0]}')

                plt.tight_layout()
                plt.savefig(f"images/mos_cer_{name_target}_fitted_sub_{name_ocr}_img{name_img}.pdf")
                # plt.show()
                plt.clf()
                plt.close()

    print(f"plotted fitted CER_comp against MOS for each distortion type, for single image (not mean)")

def plot_cer_mos_sub_mean():
    """
    Plot non-fitted CER_comp over MOS for different distortion types, mean over all images
    Subplots
    """

    data = pd.read_csv(PATHS['results_dist'])
    
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    crit = 'cer_comp'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            fig, axs = plt.subplots(3, 3, figsize=(10, 10))

            grp_dist = group_ocr.groupby('dist') 
            for idx, ((name_dist, group_dist), ax) in enumerate(zip(grp_dist, axs.flatten())):

                mean_dist = group_dist.groupby('qual')[[crit, 'mos']].mean().reset_index()

                # plot mean cer/mos over images for each quality
                im = ax.scatter(mean_dist['mos'],
                                mean_dist[crit],
                                label=group_dist['dist_name'].iloc[0],
                                marker=markers[idx],
                                cmap=colormap,
                                c=mean_dist['qual'])


                fig.colorbar(im, ax=ax,label='Quality')
                ax.set_xlabel("$\overline{MOS}$")
                ax.set_ylabel("$\overline{CER}_{comp}$")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_title(f'Dist. type: {group_dist["dist_name"].iloc[0]}')
                ax.grid()

            plt.tight_layout()
            savepath_pdf=PATHS['analyze'](f'mos_cer_{name_target}_sub_mean_{name_ocr}.pdf')
            plt.savefig(savepath_pdf)
            # savepath_png=PATHS['analyze'](f'mos_cer_{name_target}_sub_mean_{name_ocr}.png')
            # plt.savefig(savepath_png)
            plt.title(f"Comparison of CER with target {name_target} for different distortion types with {name_ocr} OCR")
            # plt.show()
            plt.clf()
            plt.close()

    print(f"plotted CER_comp against MOS for each distortion type, mean over all images")


def plot_cer_fitted_mos_sub_mean():
    """
    Plot CER_comp against different targets fitted on all images (total), for different distortion types
    Subplots
    """

    data = pd.read_csv(PATHS['results_dist'])
    
    markers = ['v', '2', 's', 'P', 'x', 'd', '.', 'p', '*']
    colormap = 'copper_r'

    crit = 'cer_comp_fitted_total'

    grp_target = data.groupby('target')
    for name_target, group_target in grp_target:
    
        grp_ocr = group_target.groupby('ocr_algo')
        for name_ocr, group_ocr in grp_ocr:

            fig, axs = plt.subplots(3, 3, figsize=(10, 10))

            grp_dist = group_ocr.groupby('dist') 
            for idx, ((name_dist, group_dist), ax) in enumerate(zip(grp_dist, axs.flatten())):

                mean_dist = group_dist.groupby('qual')[[crit, 'mos']].mean().reset_index()

                # plot mean cer/mos over images for each quality
                im = ax.scatter(mean_dist['mos'],
                                mean_dist[crit],
                                label=group_dist['dist_name'].iloc[0],
                                marker=markers[idx],
                                cmap=colormap,
                                c=mean_dist['qual'])


                fig.colorbar(im, ax=ax,label='Quality')
                ax.set_xlabel("$MOS$")
                ax.set_ylabel("$CER_{comp} (fitted)$")
                ax.set_xlim(0, 100)
                ax.set_ylim(0, 100)
                ax.set_title(f'Dist. type: {group_dist["dist_name"].iloc[0]}')
                ax.grid()

            plt.tight_layout()
            savepath_pdf=PATHS['analyze'](f'mos_cer_{name_target}_fitted_sub_mean_{name_ocr}.pdf')
            plt.savefig(savepath_pdf)
            # savepath_png=PATHS['analyze'](f'mos_cer_{name_target}_fitted_sub_mean_{name_ocr}.png')
            # plt.savefig(savepath_png)
            plt.title(f"Comparison of CER with target {name_target} for different distortion types with {name_ocr} OCR")
            # plt.show()
            plt.clf()
            plt.close()

    print(f"plotted fitted CER_comp against MOS for each distortion type, mean over all images")


############################################################################################################
# codec plots
############################################################################################################


def plot_codec_comparison_cer():
    """
    Plot CER vs quality for different codecs
    """

    # codec_config = CONFIG['codecs_config']
    # codec_config = 'scc'
    codecs_configs = ['scc', 'default']

    # load dataframe
    data = pd.read_csv(PATHS['results_codecs'])

    ocr_algos = CONFIG['ocr_algos']

    divider = 1_000_000

    # plot in same figure
    for codec_config in codecs_configs:
        for algo in ocr_algos:
            data_spec = data.loc[(data.ocr_algo == algo) & (data.codec_config == codec_config)]
            plt.plot(data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider,
                     data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_true"].mean(),
                     label='VTM true GT',
                     marker='s',
                     color='blue')
            plt.plot(data_spec.loc[(data_spec.codec == "hm")].groupby('q')["size"].mean()/divider,
                     data_spec.loc[(data_spec.codec == "hm")].groupby('q')["cer_true"].mean(),
                     label='HM true GT',
                     marker='^',
                     color='cyan')
            plt.plot(data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider,
                     data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_pseudo"].mean(),
                     label='VTM pseudo GT',
                     marker='8',
                     color='red')
            plt.plot(data_spec.loc[(data_spec.codec == "hm")].groupby('q')["size"].mean()/divider,
                     data_spec.loc[(data_spec.codec == "hm")].groupby('q')["cer_pseudo"].mean(),
                     label='HM pseudo GT',
                     marker='v',
                     color='orange')

            qvalues = data_spec.q.unique()
            xvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["size"].mean()/divider
            yvalues = data_spec.loc[(data_spec.codec == "vtm")].groupby('q')["cer_pseudo"].mean()

            for q, x, y in zip(qvalues, xvalues, yvalues):
                plt.annotate(f"QP={q}", (x, y), textcoords="offset points", xytext=(0, 10), ha='center')




            # plt.xlim(0, 0.4)
            plt.ylim(0, 0.6)
            plt.xlabel("Avg size of images in Mbit")
            plt.ylabel("Avg CER")
            plt.title(f"Comparison of codecs for {algo} with {codec_config} codec config")
            plt.grid()
            plt.legend()
            savepath_pdf=PATHS['analyze'](f'codec_comparison_{algo}_{codec_config}.pdf')
            savepath_png=PATHS['analyze'](f'codec_comparison_{algo}_{codec_config}.png')
            plt.savefig(savepath_pdf)
            plt.savefig(savepath_png)
            plt.clf()
            plt.close()
            # plt.show()

    print(f"plotted CER against quality for each codec")


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
        # plt.ylim(0, 0.35)
        plt.xlabel("Avg size of images in Mbit")
        plt.ylabel("Avg PSNR in dB")
        plt.title(f"Comparison of codecs PSNR with {codec_config} codec config")
        plt.grid()
        plt.legend()
        savepath_pdf=PATHS['analyze'](f'codec_comparison_PSNR_{codec_config}.pdf')
        savepath_png=PATHS['analyze'](f'codec_comparison_PSNR_{codec_config}.png')
        plt.savefig(savepath_pdf)
        plt.savefig(savepath_png)
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
    plot_cer_mos_sub_mean()
    plot_cer_fitted_mos_sub_mean()

    # plot codec comparison
    # plot psnr of different codecs for different qps
    # plot_codec_comparison_psnr()

    # absolute difference of pixels in images
    # image_diff()


if __name__ == '__main__':

    pass
    pipeline()
