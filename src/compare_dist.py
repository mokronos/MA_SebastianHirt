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

    # fit data once with cer and once with cer_comp
    def fit_data(group):
        group["mos_fitted"], model_params = helpers.nonlinearfitting(group["cer"], group["mos"])
        group["model_params"] = [model_params] * len(group)
        # print(f"calculated fitted single values for group {group.name}, non-comp")
        group["mos_comp_fitted"], model_params_comp = helpers.nonlinearfitting(group["cer_comp"], group["mos"])
        group["model_params_comp"] = [model_params_comp] * len(group)
        # print(f"calculated fitted single values for group {group.name}, comp")
        return group

    # add cer fitted for every image separately
    data = data.groupby(["img_num", "dist", "ocr_algo", "target"], group_keys=True).apply(fit_data)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated fitted values for every image separately")

    return data


def add_fitted_total(data):

    def fit_data(group):
        group["mos_fitted_total"], model_params = helpers.nonlinearfitting(group["cer"], group["mos"])
        group["model_params_total"] = [model_params] * len(group)
        # print(f"calculated fitted total values for group {group.name}, non-comp")
        group["mos_comp_fitted_total"], model_params_comp = helpers.nonlinearfitting(group["cer_comp"], group["mos"])
        group["model_params_comp_total"] = [model_params_comp] * len(group)
        # print(f"calculated fitted total values for group {group.name}, comp")
        return group

    # add cer fitted for every all images together
    data = data.groupby(["dist", "ocr_algo", "target"], group_keys=True).apply(fit_data)
    data.reset_index(drop=True, inplace=True)

    def fit_data_overall(group):
        group["mos_fitted_total_overall"], model_params = helpers.nonlinearfitting(group["cer"], group["mos"])
        group["model_params_total_overall"] = [model_params] * len(group)
        # print(f"calculated fitted total values for group {group.name}, non-comp")
        group["mos_comp_fitted_total_overall"], model_params_comp = helpers.nonlinearfitting(group["cer_comp"], group["mos"])
        group["model_params_comp_total_overall"] = [model_params_comp] * len(group)
        # print(f"calculated fitted total values for group {group.name}, comp")
        return group

    # add cer fitted for every all images together
    data = data.groupby(["ocr_algo", "target"], group_keys=True).apply(fit_data_overall)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated fitted values on all images together")

    return data


def add_pearson(data):

    def pearson(group):
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp"])
        group[f"pearson"] = p[0]
        # print(f"calculated pearson correlation for group {group.name}, non-fitted")

        # pearson cant deal with nan values
        # fitting sometimes fails so we skip if there are nan values
        if group[f"mos_comp_fitted_total"].isnull().values.any():
            log.info(f"cer_comp_fitted is null for {group.name[2]}")
            return group
        p = scipy.stats.pearsonr(group[f"mos"], group[f"mos_comp_fitted_total"])
        group[f"pearson_fitted"] = p[0]
        # print(f"calculated pearson correlation for group {group.name}, fitted")
        return group

    def pearson_overall(group):
        p = scipy.stats.pearsonr(group[f"mos"], group[f"cer_comp"])
        group[f"pearson_overall"] = p[0]
        # print(f"calculated pearson correlation for group {group.name}, non-fitted")

        # pearson cant deal with nan values
        # fitting sometimes fails so we skip if there are nan values
        if group[f"mos_comp_fitted_total"].isnull().values.any():
            log.info(f"cer_comp_fitted is null for {group.name[2]}")
            return group
        p = scipy.stats.pearsonr(group[f"mos"], group[f"mos_comp_fitted_total_overall"])
        group[f"pearson_fitted_overall"] = p[0]
        # print(f"calculated pearson correlation for group {group.name}, fitted")
        return group

    # calculate seperatly for distortions, algos and cer targets
    data = data.groupby(["dist", "ocr_algo", "target"], group_keys=True).apply(pearson)
    data.reset_index(drop=True, inplace=True)

    data = data.groupby(["ocr_algo", "target"], group_keys=True).apply(pearson_overall)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated pearson correlation")

    return data

def add_spearman_ranked(data):

    def spearman_ranked(group):
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp"])
        group[f"spearmanr"] = p[0]
        # print(f"calculated spearman ranked correlation for group {group.name}, non-fitted")
        p = scipy.stats.spearmanr(group[f"mos"], group[f"mos_comp_fitted_total"])
        group[f"spearmanr_fitted"] = p[0]
        # print(f"calculated spearman ranked correlation for group {group.name}, fitted")
        return group

    def spearman_ranked_overall(group):
        p = scipy.stats.spearmanr(group[f"mos"], group[f"cer_comp"])
        group[f"spearmanr_overall"] = p[0]
        # print(f"calculated spearman ranked correlation for group {group.name}, non-fitted")
        p = scipy.stats.spearmanr(group[f"mos"], group[f"mos_comp_fitted_total"])
        group[f"spearmanr_fitted_overall"] = p[0]
        # print(f"calculated spearman ranked correlation for group {group.name}, fitted")
        return group

    # calculate seperatly for distortions, algos and cer targets
    data = data.groupby(["dist", "ocr_algo", "target"], group_keys=True).apply(spearman_ranked)
    data.reset_index(drop=True, inplace=True)

    # calculate seperatly for distortions, algos and cer targets
    data = data.groupby(["ocr_algo", "target"], group_keys=True).apply(spearman_ranked_overall)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated spearman ranked correlation")

    return data

def add_rmse(data):

    def rmse(group):
        group[f"rmse"] = helpers.rmse(group[f"cer_comp"], group[f"mos"])
        group[f"rmse_fitted"] = helpers.rmse(group[f"mos_comp_fitted_total"], group[f"mos"])
        return group

    def rmse_overall(group):
        group[f"rmse_overall"] = helpers.rmse(group[f"cer_comp"], group[f"mos"])
        group[f"rmse_fitted_overall"] = helpers.rmse(group[f"mos_comp_fitted_total"], group[f"mos"])
        return group

    # calculate seperatly for distortions, algos and cer targets
    data = data.groupby(["dist", "ocr_algo", "target"], group_keys=True).apply(rmse)
    data.reset_index(drop=True, inplace=True)

    # calculate seperatly for distortions, algos and cer targets
    data = data.groupby(["ocr_algo", "target"], group_keys=True).apply(rmse_overall)
    data.reset_index(drop=True, inplace=True)

    print(f"calculated rmse")

    return data

if __name__ == "__main__":
    algos = CONFIG["ocr_algos"]
    data = setup(ids="scid_img_ids")

    print(data)
    data = add_cer(data)
    data = add_cer_comp(data)
    # data = add_fitted(data)
    data = add_fitted_total(data)
    data = add_pearson(data)
    data = add_spearman_ranked(data)
    data = add_rmse(data)

    print(data)

    data.to_csv(PATHS["results_dist"]())
    data.to_pickle(PATHS["results_dist"](ext="pkl"))
