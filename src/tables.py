import bjontegaard as bd
import pandas as pd
from config import PATHS


def get_bjontegaard(data):

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

def setup_codecs():

    data = pd.read_csv(PATHS[f"results_codecs"])
    return data

if __name__ == "__main__":

    data = setup_codecs()
    print(data)
    get_bjontegaard(data)
