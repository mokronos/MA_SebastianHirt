CONFIG = {
        # images to use for experiments
        "scid_img_ids": [1, 3, 4, 5, 29],
        # "scid_img_ids": [1, 3],

        # distortions levels to use for experiments
        "scid_dists": [1, 2, 3, 4, 5, 6, 7, 8, 9],

        # quality levels to use for experiments
        "scid_quals": [1, 2, 3, 4, 5],

        # codecs to use for experiments
        "codecs": ["vtm", "hm"],

        # images to use for codec experiments
        # "codecs_img_ids": [1],
        "codecs_img_ids": [1, 3, 4, 5, 29],

        # q's levels to use for experiments
        "codecs_qs": [35, 40, 45, 50],
        # "codecs_qs": [22, 27, 32, 37],

        # ocr algorithms to use for experiments
        "ocr_algos": ["ezocr", "tess"],
        # "ocr_algos": ["ezocr"],

        # distortion names
        # GN: Gaussian Noise, GB: Gaussian Blur, MB: Motion Blur, CC: Contrast Change
        # JPEG: JPEG, JPEG2000: JPEG2000, CSC: Color Saturation Change, HEVC-SCC: HEVC-SCC
        # CQD: Color Quantization with dithering
        "dist_names": {1: 'GN', 2: 'GB', 3: 'MB', 4: 'CC', 5: 'JPEG', 6: 'JPEG2000', 7: 'CSC', 8: 'HEVC-SCC', 9: 'CQD'}
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

        # Path to images transformed with VTM(VVC codec)
        "images_vtm":
        lambda num, q:
        f"data/raw/scid/vtm/SCI{num if num > 9 else f'0{num}'}vtm{q}.bmp",

        # Path to images transformed with HM(HEVC codec)
        "images_hm":
        lambda num, q:
        f"data/raw/scid/hm/SCI{num if num > 9 else f'0{num}'}hm{q}.bmp",

        # Path to size text file for VTM images
        "size_vtm":
        lambda num, q:
        f"data/raw/scid/vtm/SCI{num if num > 9 else f'0{num}'}vtm{q}.txt",

        # Path to size text file for HM images
        "size_hm":
        lambda num, q:
        f"data/raw/scid/hm/SCI{num if num > 9 else f'0{num}'}hm{q}.txt",

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

        # predition on images transformed with VTM(VVC codec)
        "pred_vtm":
        lambda num, q, algo = "ezocr", ext="txt":
        f"results/pred/scid/vtm/{algo}/SCI{num if num > 9 else f'0{num}'}vtm{q}.{ext}",

        # predition on images transformed with HM(HEVC codec)
        "pred_hm":
        lambda num, q, algo = "ezocr", ext="txt":
        f"results/pred/scid/hm/{algo}/SCI{num if num > 9 else f'0{num}'}hm{q}.{ext}",

        # Path to the folder containing experiment images
        "images_exp": "exp",

        # Path to the folder containing result images
        "analyze":
        lambda suff:
        f"images/analyze/{suff}",

        # Path to dataframe with distorted images results
        "results_dist":
        f"results/summaries/results_dist.csv",

        # Path to dataframe with distorted images results
        "results_spearman_pearson":
        lambda algo:
        f"results/summaries/results_dist_spear_pears_{algo}.csv",

        # Path to dataframe with codec comparison results
        "results_codecs":
        f"results/summaries/results_codecs.csv",

        # Path to the folder containing latex tables
        "latex_tables":
        lambda suff:
        f"latex/tables/{suff}.tex",
        }

        
