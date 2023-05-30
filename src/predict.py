import helpers
import logging as log
from config import PATHS, CONFIG

log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
log.disable(level=log.DEBUG)


def pred_dist():
    ocr_algos = CONFIG["ocr_algos"]

    # get paths
    load_paths = helpers.create_paths(PATHS["images_scid_dist"],
                                      CONFIG["scid_img_ids"],
                                      CONFIG["scid_dists"],
                                      CONFIG["scid_quals"])

    # run prediction
    for algo in ocr_algos:

        preds = helpers.pred_data(load_paths, algo=algo)

                    
        save_paths_csv = helpers.create_paths(PATHS["pred_dist"],
                                            CONFIG["scid_img_ids"],
                                            CONFIG["scid_dists"],
                                            CONFIG["scid_quals"],
                                            algo=algo, ext="csv")

        save_paths_txt = helpers.create_paths(PATHS["pred_dist"],
                                            CONFIG["scid_img_ids"],
                                            CONFIG["scid_dists"],
                                            CONFIG["scid_quals"],
                                            algo=algo, ext="txt")

        # save predictions
        for pred, save_path_txt, save_path_csv in zip(preds, save_paths_txt, save_paths_csv):
            pred.to_csv(save_path_csv, index=False)
            text = helpers.csv_to_text(pred)
            with open(save_path_txt, 'w') as f:
                f.write(text)

    log.info(f"done with predictions on distored images")

def pred_ref():
    """
    Predict on the reference images
    """

    ocr_algos = CONFIG["ocr_algos"]

    # get paths
    load_paths = helpers.create_paths(PATHS["images_scid_ref"],
                                      CONFIG["scid_img_ids"])

    # run prediction
    for algo in ocr_algos:

        preds = helpers.pred_data(load_paths, algo=algo)

                    
        save_paths_csv = helpers.create_paths(PATHS["pred_ref"],
                                            CONFIG["scid_img_ids"],
                                            algo=algo, ext="csv")

        save_paths_txt = helpers.create_paths(PATHS["pred_ref"],
                                            CONFIG["scid_img_ids"],
                                            algo=algo, ext="txt")

        # save predictions
        for pred, save_path_txt, save_path_csv in zip(preds, save_paths_txt, save_paths_csv):
            pred.to_csv(save_path_csv, index=False)
            text = helpers.csv_to_text(pred)
            with open(save_path_txt, 'w') as f:
                f.write(text)

    log.info(f"done with predictions on reference images")


def pred_codec(codec="vtm", config="scc"):

    ocr_algos = CONFIG["ocr_algos"]
    # ocr_algos = ["ezocr"]

    # get paths
    load_paths_vtm = helpers.create_paths(PATHS[f"images_{codec}_{config}"],
                                          CONFIG["codecs_img_ids"],
                                          CONFIG["codecs_qs"])

    for algo in ocr_algos:

        save_paths_csv = helpers.create_paths(PATHS[f"pred_{codec}_{config}"],
                                                  CONFIG["codecs_img_ids"],
                                                  CONFIG["codecs_qs"],
                                                  algo=algo, ext="csv")

        save_paths_txt = helpers.create_paths(PATHS[f"pred_{codec}_{config}"],
                                                  CONFIG["codecs_img_ids"],
                                                  CONFIG["codecs_qs"],
                                                  algo=algo, ext="txt")

        # create directories if not exist
        helpers.create_dir(save_paths_csv)
        helpers.create_dir(save_paths_txt)


        # run prediction
        preds = helpers.pred_data(load_paths_vtm, algo=algo)

        # save predictions
        for pred, save_path_txt, save_path_csv in zip(preds, save_paths_txt, save_paths_csv):
            pred.to_csv(save_path_csv, index=False)
            text = helpers.csv_to_text(pred)
            with open(save_path_txt, 'w') as f:
                f.write(text)

    log.info('done')


if __name__ == "__main__":
    # pred_dist()
    # pred_ref()
    pred_codec(codec="vtm", config="scc")
    pred_codec(codec="hm", config="scc")
