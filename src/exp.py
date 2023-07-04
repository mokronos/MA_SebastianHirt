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

def fit_example():

    subj = [30, 50, 80, 90]*10
    obj = [20, 30, 60, 80]*10

    subj += np.random.normal(-5, 5, len(subj))
    obj += np.random.normal(-5, 5, len(obj))

    beta0 = [np.max(subj), np.min(subj),
             np.mean(obj), np.std(obj)/4]
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

if __name__ == "__main__":
    pass
    # corr()
    # fitting()
    fit_example()
