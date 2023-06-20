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

    perm = itertools.product(ids, qs, codec, codec_config, ocr_algos)

    data = pd.DataFrame(perm, columns=["img_num", "q", "codec", "codec_config", "ocr_algo"])

    return data

def add_cer_true(data):
    # add cer column
    data[f"cer_true"] = data.apply(
            lambda row:
            helpers.char_error_rate(
                helpers.load_line_text(
                    PATHS["gt_scid_line"](row["img_num"])),
                helpers.load_line_text(
                    PATHS[f"pred_{row['codec']}_{row['codec_config']}"](row["img_num"],
                                                                        row["q"],
                                                                        algo=row["ocr_algo"],
                                                                        ext="txt")
                    )
                ),
            axis=1)

def add_cer_pseudo(data):
    # add cer column
    data[f"cer_pseudo"] = data.apply(
            lambda row:
            helpers.char_error_rate(
                helpers.load_line_text(
                    PATHS["pred_ref"](row["img_num"], algo=row["ocr_algo"])),
                helpers.load_line_text(
                    PATHS[f"pred_{row['codec']}_{row['codec_config']}"](row["img_num"],
                                                                        row["q"],
                                                                        algo=row["ocr_algo"],
                                                                        ext="txt")
                    )
                ),
            axis=1)

def add_size(data):

    data["size"] = data.apply(
            lambda row:
            helpers.get_size(
                PATHS[f"size_{row['codec']}_{row['codec_config']}"](row["img_num"],
                                                                    row["q"])
                ),
            axis=1)

def add_psnr(data):

    data["psnr"] = data.apply(
            lambda row:
            helpers.get_psnr(
                PATHS[f"images_scid_ref"](row["img_num"]),
                PATHS[f"images_{row['codec']}_{row['codec_config']}"](row["img_num"],
                                                                            row["q"])
                ),
            axis=1)

def add_cer_comp(data, algo="ezocr"):
    data[f"cer_comp_{algo}"] = (1 - data[f"cer_{algo}"]) * 100

if __name__ == "__main__":

    data = setup(ids="codecs_img_ids_combined")
    add_cer_true(data)
    add_cer_pseudo(data)
    add_size(data)
    add_psnr(data)

    data.to_csv(PATHS[f"results_codecs"], index=False)
    print(data)

    # slice = data.loc[
    #         (data["ocr_algo"] == "ezocr")
    #         # & (data["img_num"] == 1)
    #         & (data["codec"] == "vtm")
    #         & (data["codec_config"] == "scc")
    #         ]
    # print(slice)
