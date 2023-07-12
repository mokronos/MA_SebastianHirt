CONFIG = {
        # images to use for experiments
        # images where text is most dominant, content not distracting for MOS
        # "scid_img_ids": [1, 3, 4, 5, 29],
        "scid_img_ids": [1, 2, 3, 4, 5, 6, 7, 8, 11, 12, 15, 18, 20, 21, 24, 29],

        # distortions levels to use for experiments
        "scid_dists": [1, 2, 3, 4, 5, 6, 7, 8, 9],

        # quality levels to use for experiments
        "scid_quals": [1, 2, 3, 4, 5],

        # codecs to use for experiments
        "codecs": ["vtm", "hm"],

        # images to use for codec experiments
        # images where text is most dominant
        "codecs_img_ids": [1, 3, 4, 5, 29],
        # "codecs_img_ids": [1],
        # images with text, but also lots of other content (distracting for MOS)
        "codecs_img_ids_extra": [2, 6, 7, 8, 9, 11, 12, 13, 15, 16, 18, 19, 20,
                                 21, 22, 23, 24, 25, 27],
        # both combined
        "codecs_img_ids_combined": [1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13,
                                    15, 16, 18, 19, 20, 21, 22, 23, 24, 25,
                                    27, 29],

        # q's levels to use for experiments
        "codecs_qs": [35, 40, 45, 50],
        # "codecs_qs": [22, 27, 32, 37],

        # codec config (scc or default)
        "codecs_config": ["scc", "default"],
        # "codecs_config": ["scc"],

        # ocr algorithms to use for experiments
        "ocr_algos": ["ezocr", "tess"],
        # "ocr_algos": ["ezocr"],

        # target to use for experiments
        "targets": ["gt", "ref"],

        # distortion names
        # GN: Gaussian Noise, GB: Gaussian Blur, MB: Motion Blur, CC: Contrast Change
        # JPEG: JPEG, JPEG2000: JPEG2000, CSC: Color Saturation Change, HEVC-SCC: HEVC-SCC
        # CQD: Color Quantization with dithering
        "dist_names": {1: 'GN', 2: 'GB', 3: 'MB', 4: 'CC', 5: 'JPEG',
                       6: 'JPEG2000', 7: 'CSC', 8: 'HEVC-SCC', 9: 'CQD'}
        }


PATHS = {
        # Path to the folder containing the images
        # distorted images
        "images_scid_dist":
        lambda num, dist, qual:
        f"data/raw/scid/DistortedSCIs/SCI{num if num > 9 else f'0{num}'}_{dist}_{qual}.bmp",

        # reference images
        "images_scid_ref":
        lambda num:
        f"data/raw/scid/ReferenceSCIs/SCI{num if num > 9 else f'0{num}'}.bmp",

        # Path to images transformed with VTM(VVC codec), scc config
        "images_codec":
        lambda num, q, codec_config = "default", codec = "hm":
        f"data/raw/scid/{codec_config}/{codec}/SCI{num if num > 9 else f'0{num}'}{codec}{q}.bmp",

        # Path to size text file for VTM images scc config
        "size_codec":
        lambda num, q, codec_config = "default", codec = "hm":
        f"data/raw/scid/{codec_config}/{codec}/SCI{num if num > 9 else f'0{num}'}{codec}{q}.txt",

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
            lambda num, dist, qual, algo = "ezocr", ext="txt":
                f"results/pred/scid/dist/{algo}/SCI{num if num > 9 else f'0{num}'}_{dist}_{qual}.{ext}",

        # prediction on reference images
        "pred_ref":
            lambda num, algo = "ezocr", ext="txt":
                f"results/pred/scid/ref/{algo}/SCI{num if num > 9 else f'0{num}'}.{ext}",

        # predition on images transformed with VTM(VVC codec), scc config
        "pred_codec":
            lambda num, q, codec_config = "default", codec = "hm", algo = "ezocr", ext="txt":
                f"results/pred/scid/{codec_config}/{codec}/{algo}/SCI{num if num > 9 else f'0{num}'}{codec}{q}.{ext}",

        # Path to the folder containing experiment images
        "images_exp": "exp",

        # Path to the folder containing result images
        "analyze":
            lambda suff:
                f"images/analyze/{suff}",

        # Path to dataframe with distorted images results
        "results_dist":
            lambda ext="csv":
                f"results/summaries/results_dist.{ext}",

        # Path to dataframe with distorted images results
        "results_spearman_pearson":
            lambda suffix, ext:
                f"results/summaries/results_dist_spear_pears_{suffix}.{ext}",

        # Path to dataframe with codec comparison bjontegaard results
        "results_bjontegaard":
            lambda ext:
                f"results/summaries/results_bjontegaard.{ext}",

        # Path to dataframe with codec comparison results
        "results_codecs":
            f"results/summaries/results_codecs.csv",

        # Path to dataframe with comparison of reference and gt predictions
        "results_ref":
            f"results/summaries/results_ref.csv",

        # Path to dataframe with comparison of reference and gt predictions
        "results_ref_mean":
            lambda ext:
                f"results/summaries/results_ref_mean.{ext}",

        # Path to the folder containing latex tables
        "latex_tables":
            lambda suff:
                f"latex/tables/{suff}.tex",
        }


