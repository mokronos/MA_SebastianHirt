from config import PATHS, CONFIG
import pandas as pd
import helpers
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from scipy.optimize import curve_fit

def model(x, a, b, c, d):
    return ((a-b)/(1+np.exp((-x+c)/d)))+b

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
             np.mean(crit_data), np.std(crit_data)/4]

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

if __name__ == "__main__":
    pass
    corr()
    fitting()
