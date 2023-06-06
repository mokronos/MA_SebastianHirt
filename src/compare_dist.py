# create dataframes with all the information needed to create results
import helpers
import logging as log
import scipy.stats
from config import PATHS, CONFIG
import pandas as pd


log.basicConfig(level=log.DEBUG, format="%(asctime)s \n %(message)s")
# log.disable(level=log.DEBUG)


def setup():
    # read mos data into dataframe
    data = helpers.read_mos(PATHS["mos_scid"])

    # get configuaration of images, distorstions and qualities
    img_id = CONFIG["scid_img_ids"]
    dists = CONFIG["scid_dists"]
    quals = CONFIG["scid_quals"]

    # filter data by img_id, dist and qual
    filtered_data = data.loc[data["img_num"].isin(img_id) &
                             data["dist"].isin(dists) &
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
                                       row["dist"],
                                       row["qual"],
                                       algo = algo,
                                       ext="txt")
                    )
                ),
            axis=1)

def add_cer_comp(data, algo="ezocr"):
    data[f"cer_comp_{algo}"] = (1 - data[f"cer_{algo}"]) * 100

def add_fitted(data, algo="ezocr"):

    # loop through all images and distortions
    data_grouped = data.groupby(["img_num", "dist"])

    for name, group in data_grouped:

        fitted = helpers.nonlinearfitting(group[f"cer_{algo}"], group["mos"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"cer_fitted_{algo}"] = fitted
        
        comp_fitted = helpers.nonlinearfitting(group[f"cer_comp_{algo}"], group["mos"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"cer_comp_fitted_{algo}"] = comp_fitted

def add_fitted_total(data, algo="ezocr"):

    # loop through all images and distortions
    data_grouped = data.groupby("dist")

    for name, group in data_grouped:

        fitted = helpers.nonlinearfitting(group[f"cer_{algo}"], group["mos"])
        data.loc[(data["dist"] == name), f"cer_fitted_{algo}"] = fitted
        
        comp_fitted = helpers.nonlinearfitting(group[f"cer_comp_{algo}"], group["mos"])
        data.loc[(data["dist"] == name), f"cer_comp_fitted_{algo}"] = comp_fitted

def add_pearson(data, algo="ezocr"):
    
    # loop through all images and distortions
    data_grouped = data.groupby(["img_num", "dist"])

    for name, group in data_grouped:
        # calculate pearson correlation coefficient
        # between mos and cer_comp for the quality levels
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"pearson_{algo}"] = p[0]

        # calculate pearson correlation coefficient
        # between mos and cer_comp_fitted for the quality levels
        if group[f"cer_comp_fitted_{algo}"].isnull().values.any():
            log.info(f"cer_comp_fitted_{algo} is null")
            continue
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp_fitted_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"pearson_fitted_{algo}"] = p[0]

def add_spearman_ranked(data, algo="ezocr"):
    
    # loop through all images and distortions
    data_grouped = data.groupby(["img_num", "dist"])

    for name, group in data_grouped:
        # calculate spearman ranked correlation coefficient
        # between mos and cer_comp for the quality levels
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"spearmanr_{algo}"] = p[0]

        # calculate spearman ranked correlation coefficient
        # between mos and cer_comp_fitted for the quality levels
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp_fitted_{algo}"])
        data.loc[(data["img_num"] == name[0]) & (data["dist"] == name[1]), f"spearmanr_fitted_{algo}"] = p[0]

def create_summary(data, algo="ezocr"):

    # table with:
    # indices:
    # - spearman ranked, pearson
    # - distortions, overall 
    # values: (value used in comparison with MOS in correlation computation)
    # - CER

    fit = f"cer_fitted_{algo}"
    data_grouped = data.groupby("dist")
    cer = []
    crit = []
    dist = []
    for name, group in data_grouped:
        p = scipy.stats.pearsonr(group["mos"], group[fit])
        s = scipy.stats.spearmanr(group["mos"], group[fit])
        cer.extend([p[0], s[0]])
        crit.extend(["pearson", "spearman"])
        dist.extend([name, name])
    
    dist_names = [CONFIG["dist_names"][id] for id in dist]
    table = pd.DataFrame({"crit": crit, "dist": dist, "dist_name": dist_names, fit: cer})
    table.sort_values(by=["crit", "dist"], inplace=True, ignore_index=True)
    table = table.pivot_table(index=["crit", "dist_name"], values=[fit])

    return table


if __name__ == "__main__":
    algos = CONFIG["ocr_algos"]
    data = setup()
    

    for algo in algos:
        add_cer(data, algo=algo)
        add_cer_comp(data, algo=algo)
        add_fitted_total(data, algo=algo)
        # add_fitted(data, algo=algo)
        # add_pearson(data, algo=algo)
    #     add_spearman_ranked(data, algo=algo)
        table = create_summary(data, algo=algo).round(2)
        print(table)
        table.to_csv(PATHS["results_spearman_pearson"](algo, "csv"))
        table.to_markdown(PATHS["results_spearman_pearson"](algo, "md"))

    data.to_csv(PATHS["results_dist"])

    # show pearson, spearman for all images and distortions
    # disp = data.pivot_table(index=["img_num", "dist_names"], values=["pearson_ezocr", "pearson_fitted_ezocr", "spearmanr_ezocr", "spearmanr_fitted_ezocr",
                                                               # "pearson_tess", "pearson_fitted_tess", "spearmanr_tess", "spearmanr_fitted_tess",])

    # disp.to_csv(PATHS["results"], index=True)
   # print(disp)
