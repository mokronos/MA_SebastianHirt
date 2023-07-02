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

    data_grouped = data.groupby(["dist_name", "ocr_algo", "target"])
    table = data_grouped[['pearson', 'spearmanr']]
    table = table.mean().reset_index()
    table = pd.melt(table,
                    id_vars=["dist_name", "ocr_algo", "target"],
                    value_vars=['pearson', 'spearmanr'],
                    var_name="corr",
                    value_name="CER_comp")

    table = table.pivot(index=["corr", "dist_name"], columns=["ocr_algo", "target"])

    table = table.round(2)


    data_grouped = data.groupby(["dist_name", "ocr_algo", "target"])
    table_fitted = data_grouped[['pearson_fitted', 'spearmanr_fitted']]
    table_fitted = table_fitted.mean().reset_index()
    table_fitted = pd.melt(table_fitted,
                    id_vars=["dist_name", "ocr_algo", "target"],
                    value_vars=['pearson_fitted', 'spearmanr_fitted'],
                    var_name="corr",
                    value_name="CER_comp_fitted")

    table_fitted = table_fitted.pivot(index=["corr", "dist_name"], columns=["ocr_algo", "target"])

    table_fitted = table_fitted.round(2)

    table.to_csv(PATHS["results_spearman_pearson"]("base", "csv"))
    table.to_markdown(PATHS["results_spearman_pearson"]("base", "md"))
    table.to_latex(PATHS["results_spearman_pearson"]("base", "tex"))
    table_fitted.to_csv(PATHS["results_spearman_pearson"]("fitted", "csv"))
    table_fitted.to_markdown(PATHS["results_spearman_pearson"]("fitted", "md"))
    table_fitted.to_latex(PATHS["results_spearman_pearson"]("fitted", "tex"))

    print(table)
    print(table_fitted)


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
