import helpers
import logging as log
from config import PATHS, CONFIG

log.basicConfig(level=log.DEBUG, format='%(asctime)s \n %(message)s')
log.disable(level=log.DEBUG)


def pred():
    ocr_algos = CONFIG["ocr_algos"]

    # get paths
    load_paths = helpers.create_paths(PATHS["images_scid_dist"],
                                      CONFIG["scid_img_ids"],
                                      CONFIG["scid_comps"],
                                      CONFIG["scid_quals"])

    # run prediction
    for algo in ocr_algos:

        preds = helpers.pred_data(load_paths, algo=algo)

                    
        save_paths_csv = helpers.create_paths(PATHS["pred_dist"],
                                            CONFIG["scid_img_ids"],
                                            CONFIG["scid_comps"],
                                            CONFIG["scid_quals"],
                                            algo=algo, ext="csv")

        save_paths_txt = helpers.create_paths(PATHS["pred_dist"],
                                            CONFIG["scid_img_ids"],
                                            CONFIG["scid_comps"],
                                            CONFIG["scid_quals"],
                                            algo=algo, ext="txt")

        # save predictions
        for pred, save_path_txt, save_path_csv in zip(preds, save_paths_txt, save_paths_csv):
            pred.to_csv(save_path_csv, index=False)
            text = helpers.csv_to_text(pred)
            with open(save_path_txt, 'w') as f:
                f.write(text)

    log.info('done')


if __name__ == "__main__":
    pred()
