# create dataframes with all the information needed to create results
import helpers
import logging as log
from config import PATHS, CONFIG
import pandas as pd


log.basicConfig(level=log.DEBUG, format="%(asctime)s \n %(message)s")
# log.disable(level=log.DEBUG)


def setup(ids="scid_img_ids"):
    """
    Create dataframe with all the combinations of ids, ocr_algos
    """

    # get configuaration of images, ocr algos
    img_id = CONFIG[ids]
    ocr_algos = CONFIG["ocr_algos"]

    data = pd.DataFrame(img_id, columns=["img_num"])

    # add column for which ocr algo was used
    tmp = pd.DataFrame(columns=data.columns)
    for algo in ocr_algos:
        data["ocr_algo"] = algo
        tmp = pd.concat([tmp, data])

    data = tmp

    return data

def add_cer(data):
    """
    Add column with CER
    """

    def cer(row):
        pred = helpers.load_line_text(PATHS["pred_ref"](row["img_num"],
                                                            algo = row["ocr_algo"],
                                                            ext="txt"))

        label = helpers.load_line_text(PATHS["gt_scid_line"](row["img_num"]))

        return helpers.char_error_rate(label, pred)

    # add cer column
    data["cer"] = data.apply(cer, axis=1)

    print(f"added CER")

    return data


def add_cer_comp(data):
    """
    Add column with CER_comp
    """

    data["cer_comp"] = (1 - data["cer"]) * 100

    print(f"added CER_comp")
    
    return data


if __name__ == "__main__":

    data = setup(ids="scid_img_ids")

    data = add_cer(data)
    data = add_cer_comp(data)

    data.to_csv(PATHS["results_ref"])
