import bjontegaard as bd
import pandas as pd
from config import PATHS, CONFIG
import scipy


def get_bjontegaard():

    data = pd.read_csv(PATHS["results_codecs"])

    divider = 1_000_000
    divider = 1
    codec_config = "scc"
    algo = "ezocr"

    data[f"cer_comp_{algo}"] = (1 - data[f"cer_{algo}"]) * 100
    # x = "psnr"
    x = "cer_true"

    rate_anchor = data.loc[(data["codec"] == "hm") &
                           (data["codec_config"] == codec_config) &
                           (data["ocr_algo"] == "tess")].groupby("q")["size"].mean()/divider
    dist_anchor = data.loc[(data["codec"] == "hm") &
                           (data["codec_config"] == codec_config) &
                           (data["ocr_algo"] == "tess")].groupby("q")[x].mean()
    rate_test = data.loc[(data["codec"] == "vtm") &
                           (data["codec_config"] == codec_config) &
                           (data["ocr_algo"] == "tess")].groupby("q")["size"].mean()/divider
    dist_test = data.loc[(data["codec"] == "vtm") &
                           (data["codec_config"] == codec_config) &
                           (data["ocr_algo"] == "tess")].groupby("q")[x].mean()


    # dist_anchor[35] = 0.23
    print(dist_anchor)
    # dist_anchor = dist_anchor.sort_index(ascending=False)
    # dist_test = dist_test.sort_index(ascending=False)
    rate_anchor = [9487.76, 4593.60, 2486.44, 1358.24]
    dist_anchor = [ 40.037,  38.615,  36.845,  34.851]
    # dist_anchor = [ 40.037,  38.615,  36.845,  37.851]
    rate_test = [9787.80, 4469.00, 2451.52, 1356.24]
    dist_test = [ 40.121,  38.651,  36.970,  34.987]


    print(f"rate_anchor: \n {rate_anchor}")
    print(f"dist_anchor: \n {dist_anchor}")
    print(f"rate_test: \n {rate_test}")
    print(f"dist_test: \n {dist_test}")

    # can't calculate as the points are not strictly increasing/decreasing

    bd_rate = bd.bd_rate(rate_anchor, dist_anchor, rate_test, dist_test, method="akima")
    bd_psnr = bd.bd_psnr(rate_anchor, dist_anchor, rate_test, dist_test, method="akima")

    print(f"bd_rate: \n {bd_rate}")
    print(f"bd_psnr: \n {bd_psnr}")

    return bd_rate, bd_psnr


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

    # get_bjontegaard()

    create_summary()

    # cer_ref_gt()
