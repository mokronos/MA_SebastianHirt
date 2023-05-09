CONFIG = {
        # images to use for experiments
        # "scid_img_ids": [1, 3, 4, 5, 29],
        "scid_img_ids": [1, 3],

        # compression levels to use for experiments
        "scid_comps": [1, 2, 3, 4, 5, 6, 7, 8, 9],

        # quality levels to use for experiments
        "scid_quals": [1, 2, 3, 4, 5],

        # ocr algorithms to use for experiments
        # "ocr_algos": ["ezocr", "tess"],
        "ocr_algos": ["ezocr"],
        }


PATHS = {
        # Path to the folder containing the images
        # distorted images
        "images_scid_dist":
        lambda num, comp, qual:
        f"data/raw/scid/DistortedSCIs/SCI{num if num > 9 else f'0{num}'}_{comp}_{qual}.bmp",

        # reference images
        "images_scid_ref":
        lambda num:
        f"data/raw/scid/ReferenceSCIs/SCI{num if num > 9 else f'0{num}'}.bmp",

        # Path to MOS txt file for scid dataset
        "mos_scid":
        "data/raw/scid/MOS_SCID.txt",

        # Path to the folder containing the ground truth
        # formatted by line, disregarding paragraphs
        "gt_scid_line":
        lambda num:
        f"data/gt/scid/line/SCI{num if num > 9 else f'0{num}'}_gt.txt",
        # formatted by paragraph, disregarding lines
        "gt_scid_para":
        lambda num:
        f"data/gt/scid/para/SCI{num if num > 9 else f'0{num}'}_gt.txt",

        # Path to the folder containing the predictions
        # prediction on distorted images
        "pred_dist":
        lambda num, comp, qual, algo = "ezocr", ext="txt":
        f"results/pred/scid/dist/{algo}/SCI{num if num > 9 else f'0{num}'}_{comp}_{qual}.{ext}",

        # prediction on reference images
        "pred_ref":
        lambda num, algo = "ezocr", ext="txt":
        f"results/pred/scid/ref/{algo}/SCI{num if num > 9 else f'0{num}'}.{ext}",

        # Path to the folder containing experiment images
        "images_exp": "exp",
        # Path to the folder containing result images
        "analyze":
        lambda suff:
        f"images/analyze/{suff}",
        }
