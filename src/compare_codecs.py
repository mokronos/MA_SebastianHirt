import pandas as pd
import itertools
import helpers
from config import PATHS, CONFIG


def setup(ids="codecs_img_ids"):
    """
    Read image ids into dataframe
    """

    ids = CONFIG[ids]
    qs = CONFIG["codecs_qs"]
    codec = CONFIG["codecs"]
    ocr_algos = CONFIG["ocr_algos"]
    codec_config = CONFIG["codecs_config"]
    targets = CONFIG["targets"]

    perm = itertools.product(ids, qs, codec, codec_config, ocr_algos, targets)

    data = pd.DataFrame(perm, columns=["img_num", "q", "codec", "codec_config", "ocr_algo", "target"])

    return data

def add_cer(data):

    def cer(row):
        pred = helpers.load_line_text(
                PATHS[f"pred_codec"](row["img_num"],
                                     row["q"],
                                     codec_config=row["codec_config"],
                                     codec=row["codec"],
                                     algo=row["ocr_algo"],
                                     ext="txt")
                )

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


def add_size(data):

    data["size"] = data.apply(
            lambda row:
            helpers.get_size(
                PATHS[f"size_codec"](row["img_num"],
                                     row["q"],
                                     codec_config=row["codec_config"],
                                     codec=row["codec"]
                                     )
                ),
            axis=1)

    return data

def add_psnr(data):

    data["psnr"] = data.apply(
            lambda row:
            helpers.get_psnr(
                PATHS[f"images_scid_ref"](row["img_num"]),
                PATHS[f"images_codec"](row["img_num"],
                                       row["q"],
                                       codec_config=row["codec_config"],
                                       codec=row["codec"])
                ),
            axis=1)

    return data


if __name__ == "__main__":

    data = setup(ids="codecs_img_ids_combined")
    print(data)
    data = add_cer(data)
    data = add_cer_comp(data)
    data = add_size(data)
    data = add_psnr(data)

    print(data)
    # data.to_csv(PATHS[f"results_codecs"], index=False)
