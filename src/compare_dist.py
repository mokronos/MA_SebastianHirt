# create dataframes with all the information needed to create results
import helpers
import logging as log
import scipy.stats
from config import PATHS, CONFIG
import pandas as pd


log.basicConfig(level=log.DEBUG, format="%(asctime)s \n %(message)s")
# log.disable(level=log.DEBUG)


def setup(ids="scid_img_ids"):
    # read mos data into dataframe
    data = helpers.read_mos(PATHS["mos_scid"])

    # get configuaration of images, distorstions and qualities
    img_id = CONFIG[ids]
    dists = CONFIG["scid_dists"]
    quals = CONFIG["scid_quals"]
    ocr_algos = CONFIG["ocr_algos"]
    targets = CONFIG["targets"]

    # add column for which ocr algo was used
    tmp = pd.DataFrame(columns=data.columns)
    for algo in ocr_algos:
        data["ocr_algo"] = algo
        tmp = pd.concat([tmp, data])

    data = tmp

    # add column for which target to calculate the CER was used (gt or ref)
    tmp = pd.DataFrame(columns=data.columns)
    for target in targets:
        data["target"] = target
        tmp = pd.concat([tmp, data])

    data = tmp

    # filter data by img_id, dist and qual defined in config (only keep relevant data)
    # and only the data we have a ground truth for
    filtered_data = data.loc[data["img_num"].isin(img_id) &
                             data["dist"].isin(dists) &
                             data["qual"].isin(quals)]

    filtered_data.reset_index(inplace=True, drop=True)

    return filtered_data

def add_cer(data):

    def cer(row):
        pred = helpers.load_line_text(PATHS["pred_dist"](row["img_num"],
                                                         row["dist"],
                                                         row["qual"],
                                                         algo = row["ocr_algo"],
                                                         ext="txt"))

        # get label either from prediction on reference image or from ground truth
        if row["target"] == "ref":
            label = helpers.load_line_text(PATHS["pred_ref"](row["img_num"],
                                                             algo = row["ocr_algo"],
                                                             ext="txt"))
        elif row["target"] == "gt":
            label = helpers.load_line_text(PATHS["gt_scid_line"](row["img_num"]))

        else:
            raise ValueError("target must be gt or ref")

        return helpers.char_error_rate(label, pred)

    # add cer column
    data["cer"] = data.apply(cer, axis=1)

    print(f"added CER")

    return data


def add_cer_comp(data):
    data["cer_comp"] = (1 - data["cer"]) * 100

    print(f"added CER_comp")
    
    return data


def add_fitted(data):

    def fit_data(group):
        group["cer_fitted"] = helpers.nonlinearfitting(group["cer"], group["mos"])
        group["cer_comp_fitted"] = helpers.nonlinearfitting(group["cer_comp"], group["mos"])
        return group

    # add cer fitted for every image separately
    data = data.groupby(["img_num", "dist", "ocr_algo"], group_keys=True).apply(fit_data)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated fitted values for every image separately")

    return data


def add_fitted_total(data):

    def fit_data(group):
        group["cer_fitted_total"] = helpers.nonlinearfitting(group["cer"], group["mos"])
        group["cer_comp_fitted_total"] = helpers.nonlinearfitting(group["cer_comp"], group["mos"])
        return group

    # add cer fitted for every all images together
    data = data.groupby(["dist", "ocr_algo"], group_keys=True).apply(fit_data)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated fitted values on all images together")

    return data


def add_pearson(data):

    def pearson(group):
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp"])
        group[f"pearson"] = p[0]

        # pearson cant deal with nan values
        # fitting sometimes fails so we skip if there are nan values
        if group[f"cer_comp_fitted"].isnull().values.any():
            log.info(f"cer_comp_fitted is null for {group.name[2]}")
            return group
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp_fitted"])
        group[f"pearson_fitted"] = p[0]
        return group

    data = data.groupby(["img_num", "dist", "ocr_algo", "target"], group_keys=True).apply(pearson)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated pearson correlation")

    return data

def add_spearman_ranked(data):

    def spearman_ranked(group):
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp"])
        group[f"spearmanr"] = p[0]
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp_fitted"])
        group[f"spearmanr_fitted"] = p[0]
        return group

    data = data.groupby(["img_num", "dist", "ocr_algo", "target"], group_keys=True).apply(spearman_ranked)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated spearman ranked correlation")

    return data


if __name__ == "__main__":
    algos = CONFIG["ocr_algos"]
    data = setup(ids="scid_img_ids")

    print(data)
    data = add_cer(data)
    data = add_cer_comp(data)
    data = add_fitted(data)
    data = add_fitted_total(data)
    data = add_pearson(data)
    data = add_spearman_ranked(data)

    # print(data[['img_num', 'mos', 'cer_comp', 'cer_comp_fitted', 'spearmanr']])

    data.to_csv(PATHS["results_dist"])
