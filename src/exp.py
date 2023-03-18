import numpy as np
from scipy.stats import spearmanr, pearsonr
from scipy.optimize import curve_fit


def model(x, a, b, c, d):
    return ((a-b)/(1+np.exp((-x+c)/d)))+b


def nonlinearfitting(objvals, subjvals):
    # calculate SROCC before the non-linear mapping
    srocc, _ = spearmanr(objvals, subjvals)

    # initialize the parameters used by the nonlinear fitting function
    beta0 = [np.max(subjvals), np.min(subjvals),
             np.mean(objvals), np.std(objvals)/4]

    # fitting a curve using the data
    max_nfev = 400
    betam, _ = curve_fit(model, objvals, subjvals, p0=beta0, method='lm',
                         maxfev=max_nfev)

    # plot objective-subjective score pairs
    # import matplotlib.pyplot as plt
    # t = np.linspace(min(objvals), max(objvals), 200)
    # ypre = model(t, *betam)
    # plt.plot(objvals, subjvals, '+b', linewidth=1)
    # plt.plot(t, ypre, 'k', linestyle='-', linewidth=2)
    # plt.show()

    # given an objective value,
    # predict the corresponding MOS (ypre) using the fitted curve
    ypre = model(np.array(objvals), *betam)

    plcc, _ = pearsonr(subjvals, ypre)  # pearson linear coefficient
    return srocc, plcc, ypre


# load data
sq = [7, 27, 2, 50, 28, 29, 20, 12, 6, 17]
q = [106, 100, 86, 101, 99, 103, 97, 113, 112, 110]

# call the nonlinearfitting function
srocc, plcc, ypre = nonlinearfitting(q, sq)
print(f'SROCC: {srocc}')
print(f'PLCC: {plcc}')
print(f'ypre: {ypre}')
