from config import PATHS, CONFIG
import pandas as pd
import helpers
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit
import cv2
import matplotlib.patches as patches

def model(x, a, b, c, d, e):
    return (a * (1/2 - (1/(1 + np.exp(b * (x - c))))) + d * x + e)

def fitting():

    # load data
    data = pd.read_pickle(PATHS["results_dist"](ext="pkl"))

    algo = "tess"
    ids = [1,2,3]
    target = "gt"
    dist = 7
    crit = "cer_comp"
    max_nfev = 0
    # filter data
    # data = data.loc[data["img_num"].isin(ids)]
    data = data.loc[data["ocr_algo"] == algo]
    data = data.loc[data["target"] == target]
    data = data.loc[data["dist"] == dist]

    mos = data["mos"].values
    crit_data = data[crit].values
    t = np.arange(crit_data.min(), crit_data.max(), 0.001)

    # fit
    # initialize the parameters used by the nonlinear fitting function
    beta0 = [np.max(mos), np.min(mos),
             np.mean(crit_data), np.std(crit_data)/4,
             0]

    # fitting a curve using the data
    params, _ = curve_fit(model, crit_data, mos, p0=beta0, method='lm',
                         maxfev=max_nfev, ftol=1.5e-08, xtol=1.5e-08)

    mos_fitted = model(np.array(crit_data), *params)

    curve = model(t, *params)
    curve_init = model(t, *beta0)
    grp = data.groupby("qual")

    a, b, c, d = params

    print(f" a: {a} b: {b} c: {c} d: {d}")
    print(f" exp: {np.exp((-1+c)/d)}")
    print(f" exp-value: {(-1+c)/d}")

    test_val = -5000.2
    test_val = np.array([test_val], dtype=np.float64)
    print(f" exp-test: {np.exp(test_val)}")

    # plot
    plt.plot(mos, crit_data, 'o', label='data')
    plt.plot(mos_fitted, crit_data, 'o', label='fitted')
    plt.plot(curve, t, label='model')
    plt.plot(curve_init, t, label='model_init')
    # plt.plot(t, curve, label='model_wrong')

    mos_means = []
    cer_comp_means = []
    for name, group in grp:
        mos_mean = group["mos"].mean()
        cer_comp_mean = group[crit].mean()
        # plt.plot(mos_mean, cer_comp_mean, 'o', label=name)
        mos_means.append(mos_mean)
        cer_comp_means.append(cer_comp_mean)


    print(f"spearmanr: {scipy.stats.spearmanr(mos_means, cer_comp_means)}")
    print(f"pearsonr: {scipy.stats.pearsonr(mos_means, cer_comp_means)}")
    print(f"spearmanr curve: {scipy.stats.spearmanr(curve, t)}")
    print(f"pearsonr curve: {scipy.stats.pearsonr(curve, t)}")
    print(f"corrcoef: {np.corrcoef(mos_means, cer_comp_means)}")
    print(f"corrcoef curve: {np.corrcoef(curve, t)}")
    plt.xlim(0, 100)
    # plt.ylim(0, 100)
    plt.xlabel("MOS")
    plt.ylabel(crit)
    plt.grid()
    plt.legend()
    plt.show()
    plt.close()

def fit_example():

    subj = [30, 50, 80, 90]*10
    obj = [20, 30, 60, 80]*10

    subj += np.random.normal(-5, 5, len(subj))
    obj += np.random.normal(-5, 5, len(obj))

    beta0 = [np.max(subj), np.min(subj),
             np.mean(obj), np.std(obj)/4,
             1]
    MAXFEV = 0
    
    # fitting a curve using the data
    params, _ = curve_fit(model, obj, subj, method='lm', p0=beta0,
                            maxfev=MAXFEV, ftol=1.5e-08, xtol=1.5e-08)

    t = np.arange(0, 100, 0.001)
    curve = model(t, *params)
    curve_init = model(t, *beta0)
    subj_fit = model(np.array(obj), *params)
    p_r = scipy.stats.pearsonr(subj, obj)[0]
    p_s = scipy.stats.spearmanr(subj, obj)[0]
    p_r_fit = scipy.stats.pearsonr(subj_fit, obj)[0]
    p_s_fit = scipy.stats.spearmanr(subj_fit, obj)[0]

    plt.plot(subj, obj, 'o', label='data')
    plt.plot(subj_fit, obj, 'o', label='fitted')
    plt.plot(curve_init, t, label='model_init')
    plt.plot(curve, t, label='model')
    plt.xlim(0, 100)
    plt.ylim(0, 100)
    plt.ylabel("objective value")
    plt.xlabel("subjective value")
    text = f"pearsonr: {p_r:.2f}\nspearmanr: {p_s:.2f}\npearsonr_fit: {p_r_fit:.2f}\nspearmanr_fit: {p_s_fit:.2f}"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)

    plt.text(0.02, 0.7, text, transform=plt.gca().transAxes,
            verticalalignment='top', bbox=props)

    # draw lines from data points to fitted points
    for i in range(len(subj)):
        plt.plot([subj[i], subj_fit[i]], [obj[i], obj[i]], 'k--', alpha=0.2)

    plt.tight_layout()
    plt.grid()
    plt.legend()
    # plt.savefig("exp/fit_example.pdf")
    # plt.savefig("exp/fit_example.png")
    plt.show()
    plt.close()



def corr():

    data = pd.read_pickle(PATHS["results_dist"](ext="pkl"))

    id = 4
    dist = 7
    algo = "tess"
    target = "gt"
    crit = "cer"
    # crits = ["mos", "cer_comp"]
    crits = ["mos_comp_fitted_total", "cer_comp"]

    # filter data
    data = data.loc[data["img_num"] == id]
    data = data.loc[data["ocr_algo"] == algo]
    data = data.loc[data["target"] == target]
    data = data.loc[data["dist"] == dist]
    print(data[crits])

def tess_order():
    """
    check how to order tesseract results that the text is in lines
    """

    id = 5
    algo = "ezocr"
    algo = "tess"

    save_paths_csv = helpers.create_paths(PATHS["pred_ref"],
                                            [id],
                                            algo=algo, ext="csv")

    save_path = save_paths_csv[0]
    print(save_path)

    pred_csv = pd.read_csv(save_path)
    print(pred_csv)

    # clean data
    pred_csv = pred_csv.loc[pred_csv["text"].str.strip() != ""]

    pred_csv["width"] = pred_csv["right"] - pred_csv["left"]
    pred_csv["height"] = pred_csv["bottom"] - pred_csv["top"]
    pred_csv.reset_index(inplace=True, drop=True)

    # sort
    # get top most box, then get left most box
    # then check if there is a box to the left in 70% of the height of the box

    def check_overlap(edge1, edge2):

        if (edge1[0] <= edge2[0] <= edge1[1] or edge1[0] <= edge2[1] <= edge1[1] or
            edge2[0] <= edge1[0] <= edge2[1] or edge2[0] <= edge1[1] <= edge2[1]):
            return True
        else:
            return False


    def sort_boxes(data):

        TOL = 0.7
        new_data = pd.DataFrame(columns=data.columns)

        tolerance = None
        while len(data) > 0:

            if tolerance:

                # get boxes with edge overlap
                in_tolerance = data.loc[data["top"].apply(lambda x: check_overlap(tolerance, (x, x + data["height"].values[0])))]

                if len(in_tolerance) > 0:
                    left = in_tolerance.loc[in_tolerance["left"] == in_tolerance["left"].min()]
                
                else:
                    # get top left most box in next iteration
                    tolerance = None
                    continue

            else:
                # get top most box
                top = data.loc[data["top"] == data["top"].min()]
                # get left most box
                left = top.loc[top["left"] == top["left"].min()]

            # add to new data
            new_data = pd.concat([new_data, left])

            # get tolerance
            tolerance = (left["top"].values[0], int(left["top"].values[0] + left["height"].values[0] * TOL))

            # remove from data
            data = data.drop(left.index)

        new_data.reset_index(inplace=True, drop=True)

        return new_data

    new_data = sort_boxes(pred_csv)

    # load image
    img_path = PATHS["images_scid_ref"](num=id)

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img)

    # draw text boxes
    for i, row in new_data.iterrows():
        x, y, w, h = row["left"], row["bottom"], row["right"]-row["left"], row["top"]-row["bottom"]
        text = row["text"] + f" ({i})"
        plt.gca().add_patch(patches.Rectangle((x, y), w, h, fill=False, color="green"))
        plt.text(x, y+10, text, color="green")
    plt.show()


def model_vis():

    beta0 = [-100, 1, 1, 0.5, 50]
    t = np.linspace(0, 100, 1000)
    y = helpers.model(t, *beta0)

    plt.plot(t, y)
    plt.show()
    

if __name__ == "__main__":
    pass
    # corr()
    # tess_order()
    # fitting()
    # fit_example()
    model_vis()
