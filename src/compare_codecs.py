from config import PATHS, CONFIG
import pandas as pd
import itertools
import logging as log
import helpers


def setup():
    """
    Read image ids into dataframe
    """

    ids = CONFIG["codecs_img_ids"]
    qs = CONFIG["codecs_qs"]
    codec = CONFIG["codecs"]
    ocr_algos = CONFIG["ocr_algos"]
    perm = itertools.product(ids, qs, codec, ocr_algos)

    data = pd.DataFrame(perm, columns=["img_num", "q", "codec", "ocr_algo"])

    return data

def add_cer_true(data):
    # add cer column
    data[f"cer_true"] = data.apply(
            lambda row:
            helpers.char_error_rate(
                helpers.load_line_text(
                    PATHS["gt_scid_line"](row["img_num"])),
                helpers.load_line_text(
                    PATHS[f"pred_{row['codec']}"](row["img_num"],
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
                    PATHS[f"pred_{row['codec']}"](row["img_num"],
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
                PATHS[f"pred_{row['codec']}"](row["img_num"],
                                             row["q"])
                ),
            axis=1)
    

if __name__ == "__main__":

    data = setup()
    add_cer_true(data)
    add_cer_pseudo(data)
    # add_size(data)
    data.to_csv(PATHS["results_codecs"], index=False)
    print(data)
