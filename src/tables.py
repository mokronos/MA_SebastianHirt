import bjontegaard as bd
import pandas as pd
from config import PATHS, CONFIG
import scipy
import matplotlib.pyplot as plt


def bjontegaard():

    data = pd.read_csv(PATHS["results_codecs"])
    print(data)

    # divider does not matter
    divider = 1_000_000
    divider = 1

    # set anchor and test codec
    codec_anchor = "hm"
    codec_test = "vtm"

    # set metric to be evaluated
    crit = "cer_comp"

    # loop over all combinations of:
    codec_configs = ["default", "scc"]
    ocr_algos = ["ezocr", "tess"]

    summary_table = pd.DataFrame(columns=["ocr_algo", "codec_config", "bd_rate_pseudo", "bd_rate_true", "diff"])

    for ocr_algo in ocr_algos:
        for codec_config in codec_configs:
                    
            target = "ref"
            rate_anchor_pseudo = data.loc[(data["codec"] == codec_anchor) &
                                   (data["codec_config"] == codec_config) &
                                   (data["ocr_algo"] == ocr_algo) &
                                   (data["target"] == target)].groupby("q")["size"].mean()/divider
            dist_anchor_pseudo = data.loc[(data["codec"] == codec_anchor) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")[crit].mean()
            rate_test_pseudo = data.loc[(data["codec"] == codec_test) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")["size"].mean()/divider
            dist_test_pseudo = data.loc[(data["codec"] == codec_test) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")[crit].mean()

            target = "gt"
            rate_anchor_true = data.loc[(data["codec"] == codec_anchor) &
                                   (data["codec_config"] == codec_config) &
                                   (data["ocr_algo"] == ocr_algo) &
                                   (data["target"] == target)].groupby("q")["size"].mean()/divider
            dist_anchor_true = data.loc[(data["codec"] == codec_anchor) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")[crit].mean()
            rate_test_true = data.loc[(data["codec"] == codec_test) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")["size"].mean()/divider
            dist_test_true = data.loc[(data["codec"] == codec_test) &
                                    (data["codec_config"] == codec_config) &
                                    (data["ocr_algo"] == ocr_algo) &
                                    (data["target"] == target)].groupby("q")[crit].mean()



            # can't calculate as the points are not strictly increasing/decreasing
            try:
                bd_rate_pseudo = bd.bd_rate(rate_anchor_pseudo, dist_anchor_pseudo,
                                            rate_test_pseudo, dist_test_pseudo,
                                            method='akima')
            except Exception as e:
                bd_rate_pseudo = None
                print(f"bd_rate_pseudo calculation failed for ocralgo: {ocr_algo}, codec: {codec_anchor}, config: {codec_config}, target: {target}")
                print(e)

            try:
                bd_rate_true = bd.bd_rate(rate_anchor_true, dist_anchor_true,
                                            rate_test_true, dist_test_true,
                                            method='akima')
            except Exception as e:
                bd_rate_true = None
                print(f"bd_rate_true calculation failed for ocralgo: {ocr_algo}, codec: {codec_anchor}, config: {codec_config}, target: {target}")
                print(e)


            print(f"config: {codec_config}, ocr_algo: {ocr_algo}")
            print(f"bd_rate_pseudo: {bd_rate_pseudo}")
            print(f"bd_rate_true: {bd_rate_true}")
            if bd_rate_pseudo is not None and bd_rate_true is not None:
                diff = bd_rate_pseudo - bd_rate_true
                print(f"bd_rate_diff: {diff}")
            else:
                diff = None
                print(f"bd_rate_diff: {diff}")

            print("*"*20)

            # append to table
            tmp = [{"ocr_algo": ocr_algo, "codec_config": codec_config,
                    "bd_rate_pseudo": bd_rate_pseudo, "bd_rate_true": bd_rate_true,
                    "diff": diff}]

            summary_table = pd.concat([summary_table, pd.DataFrame.from_records(tmp)], ignore_index=True)

    print(summary_table)
    # save table as tex and csv
    summary_table.to_csv(PATHS["results_bjontegaard"](ext="csv"), index=False)
    summary_table.style.hide(axis="index").to_latex(PATHS["results_bjontegaard"](ext="tex"))

def create_summary():

    # table with:
    # indices:
    # - spearman ranked, pearson
    # - distortions, overall 
    # values: (value used in comparison with MOS in correlation computation)
    # - CER

    data = pd.read_pickle(PATHS["results_dist"](ext="pkl"))

    # only interested in cer vs reference image
    data = data.loc[data["target"] == "ref"]

    data_grouped = data.groupby(["dist_name", "ocr_algo"])
    table = data_grouped[['spearmanr', 'pearson_fitted', "rmse_fitted"]]
    table = table.mean().reset_index()
    table = pd.melt(table,
                    id_vars=["dist_name", "ocr_algo"],
                    value_vars=['spearmanr', 'pearson_fitted', "rmse_fitted"],
                    var_name="crit",
                    value_name="CER_comp")


    table = table.pivot(index=["crit", "dist_name"], columns=["ocr_algo"])

    table = table.round(2)

    spear_overall_tess = data.loc[data['ocr_algo'] == 'tess']['spearmanr_overall'].mean().round(2)
    spear_overall_ezocr = data.loc[data['ocr_algo'] == 'ezocr']['spearmanr_overall'].mean().round(2)
    pearson_overall_tess = data.loc[data['ocr_algo'] == 'tess']['pearson_fitted_overall'].mean().round(2)
    pearson_overall_ezocr = data.loc[data['ocr_algo'] == 'ezocr']['pearson_fitted_overall'].mean().round(2)
    rmse_overall_tess = data.loc[data['ocr_algo'] == 'tess']['rmse_fitted_overall'].mean().round(2)
    rmse_overall_ezocr = data.loc[data['ocr_algo'] == 'ezocr']['rmse_fitted_overall'].mean().round(2)

    # add overall values
    table.loc[("spearmanr", "Overall"), ("CER_comp", "tess")] = spear_overall_tess
    table.loc[("spearmanr", "Overall"), ("CER_comp", "ezocr")] = spear_overall_ezocr
    table.loc[("pearson_fitted", "Overall"), ("CER_comp", "tess")] = pearson_overall_tess
    table.loc[("pearson_fitted", "Overall"), ("CER_comp", "ezocr")] = pearson_overall_ezocr
    table.loc[("rmse_fitted", "Overall"), ("CER_comp", "tess")] = rmse_overall_tess
    table.loc[("rmse_fitted", "Overall"), ("CER_comp", "ezocr")] = rmse_overall_ezocr

    table = table.sort_index(level=0, ascending=True)

    table.to_csv(PATHS["results_spearman_pearson"]("base", "csv"))
    table.to_markdown(PATHS["results_spearman_pearson"]("base", "md"))
    table.to_latex(PATHS["results_spearman_pearson"]("base", "tex"))

    print(table)
    print(f"spearmanr for tess overall: {spear_overall_tess}")
    print(f"spearmanr for ezocr overall: {spear_overall_ezocr}")
    print(f"pearson for tess overall: {pearson_overall_tess}")
    print(f"pearson for ezocr overall: {pearson_overall_ezocr}")
    print(f"rmse for tess overall: {rmse_overall_tess}")
    print(f"rmse for ezocr overall: {rmse_overall_ezocr}")

def cer_ref_gt():
    """
    Table with mean CER (CER_comp) for gt vs ref, for each ocr_algo
    """

    data = pd.read_csv(PATHS["results_ref"])

    print(data)

    data_grouped = data.groupby("ocr_algo")

    table = data_grouped[["cer", "cer_comp"]].mean()
    print(table)

    table.to_csv(PATHS["results_ref_mean"]("csv"))
    table.to_markdown(PATHS["results_ref_mean"]("md"))


if __name__ == "__main__":

    bjontegaard()

    # create_summary()

    # cer_ref_gt()
