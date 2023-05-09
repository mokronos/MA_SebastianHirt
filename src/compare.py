# create dataframes with all the information needed to create results
import helpers
import logging as log
import scipy.stats
from config import PATHS, CONFIG

log.basicConfig(level=log.DEBUG, format="%(asctime)s \n %(message)s")
log.disable(level=log.DEBUG)


def setup():
    # read mos data into dataframe
    data = helpers.read_mos(PATHS["mos_scid"])

    # get configuaration of images, compression and quality
    img_id = CONFIG["scid_img_ids"]
    comps = CONFIG["scid_comps"]
    quals = CONFIG["scid_quals"]

    # filter data by img_id, comp and qual
    filtered_data = data.loc[data["img_num"].isin(img_id) &
                             data["comp"].isin(comps) &
                             data["qual"].isin(quals)].reset_index(drop=True)

    return filtered_data

def add_cer(data, algo="ezocr"):
    # add cer column
    data[f"cer_{algo}"] = data.apply(
            lambda row:
            helpers.char_error_rate(
                helpers.load_line_text(
                    PATHS["gt_scid_line"](row["img_num"])),
                helpers.load_line_text(
                    PATHS["pred_dist"](row["img_num"],
                                       row["comp"],
                                       row["qual"],
                                       algo = algo,
                                       ext="txt")
                    )
                ),
            axis=1)

def add_cer_comp(data, algo="ezocr"):
    data[f"cer_comp_{algo}"] = (1 - data[f"cer_{algo}"]) * 100

def add_fitted(data, algo="ezocr"):

    # loop through all images and compression
    data_grouped = data.groupby(["img_num", "comp"])

    for name, group in data_grouped:

        fitted = helpers.nonlinearfitting(group[f"cer_{algo}"], group["mos"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"cer_fitted_{algo}"] = fitted
        
        comp_fitted = helpers.nonlinearfitting(group[f"cer_comp_{algo}"], group["mos"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"cer_comp_fitted_{algo}"] = comp_fitted


def add_pearson(data, algo="ezocr"):
    
    # loop through all images and compression
    data_grouped = data.groupby(["img_num", "comp"])

    for name, group in data_grouped:
        # calculate pearson correlation coefficient
        # between mos and cer_comp for the quality levels
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"pearson_{algo}"] = p[0]

        # calculate pearson correlation coefficient
        # between mos and cer_comp_fitted for the quality levels
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp_fitted_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"pearson_fitted_{algo}"] = p[0]

def add_spearman_ranked(data, algo="ezocr"):
    
    # loop through all images and compression
    data_grouped = data.groupby(["img_num", "comp"])

    for name, group in data_grouped:
        # calculate spearman ranked correlation coefficient
        # between mos and cer_comp for the quality levels
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"spearmanr_{algo}"] = p[0]

        # calculate spearman ranked correlation coefficient
        # between mos and cer_comp_fitted for the quality levels
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp_fitted_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["comp"] == name[1]), f"spearmanr_fitted_{algo}"] = p[0]


if __name__ == "__main__":
    algos = CONFIG["ocr_algos"]
    data = setup()

    for algo in algos:
        print(data)
        add_cer(data, algo=algo)
        print(data)
        add_cer_comp(data, algo=algo)
        print(data)
        add_fitted(data, algo=algo)
        print(data)
        add_pearson(data, algo=algo)
        print(data)
        add_spearman_ranked(data, algo=algo)
        print(data)

    # data.to_csv(PATHS["results_scid"], index=False)

    # show pearson, spearman for all images and compressions
    disp = data.pivot_table(index=["img_num", "comp"], values=["pearson_ezocr", "pearson_fitted_ezocr", "spearmanr_ezocr", "spearmanr_fitted_ezocr"])
    print(disp)
