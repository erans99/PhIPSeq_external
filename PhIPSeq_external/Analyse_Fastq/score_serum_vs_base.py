import math
import os
import time
from decimal import *

import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.optimize
from sklearn import linear_model

PRECISION = 200
MAX_SUM = 10 ** PRECISION

MIN_BASE = 25  # given our 50M reads of 344,000 oligos, then the average is about 173,
# so we will treat all under 25 as base level.
MIN_CNT_BASE_CELL = 100
DROP = 0.05

MAX_LAM = 0.99
MIN_LAM = 0.01
MIM_VALS_TO_FIT = 5

numpy.seterr(all='raise')


def read_file(f, cols):
    found = pandas.read_csv(f, index_col=0)
    found['good'] = 0
    for c in cols:
        found['good'] += found[c]
    return found[['good']]


def ML_func(lam, ser):
    mean = 1. * ser.sum() / len(ser)
    try:
        return ((ser * (ser - 1)) / (mean + (ser - mean) * lam)).sum() - len(ser) * mean
    except:
        return -MAX_SUM


def ML_func_prime(lam, ser):
    mean = 1. * ser.sum() / len(ser)
    try:
        return -((ser * (ser - 1) * (ser - mean)) / ((mean + (ser - mean) * lam) ** 2)).sum()
    except:
        return -MAX_SUM


def match_GP(inds, found, col, drop, outf=None):
    MAX_SC = 100
    f = found.reindex(index=inds, columns=[col], fill_value=0)
    f.sort_values(col, inplace=True)
    sample = f[col].values
    num_drop = int(len(sample) * drop)
    if num_drop > 0:
        sample = numpy.concatenate((sample[:-num_drop], sample[-2 * num_drop:-num_drop]))
    try:
        lam = scipy.optimize.newton(ML_func, 0.5, ML_func_prime, args=(sample,))
        best = [lam, math.fabs(ML_func(lam, sample)), (1. * sample.sum() / len(sample))]
    except:
        best = [-1, MAX_SC, -1]
    if outf is None:
        if best[1] >= MAX_SC:
            print("   %d points (%d 0s&1s) in dist, failed" % (len(sample), sum(sample < 2)))
        else:
            print("   %d points (%d 0s&1s) in dist, lambda %g (score %g) theta %g (base %g)" %
                  (len(sample), sum(sample < 2), best[0], best[1], best[2] * (1 - best[0]), best[2]))
    else:
        if best[1] >= MAX_SC:
            outf.write("   %d points (%d 0s&1s) in dist, failed\n" % (len(sample), sum(sample < 2)))
        else:
            outf.write("   %d points (%d 0s&1s) in dist, lambda %g (score %g) theta %g (base %g)\n" %
                       (len(sample), sum(sample < 2), best[0], best[1], best[2] * (1 - best[0]), best[2]))
    if best[1] >= MAX_SC:
        return -1, MAX_SC, -1
    return best[0], best[1], best[2]


def count_winds(vals, wind):
    res = {}
    for i, w in enumerate(range(0, int(vals.index.max()), wind)):
        try:
            res[i] = vals.loc[range(w, w + wind)].sum()
        except:
            res[i] = 0
    return pandas.Series(res)


def match_GPs(all_empty, found, outf=None, col='no_err', drop=DROP, th=MIN_CNT_BASE_CELL, wind=1, f_txt=None):
    if f_txt is None:
        if outf is not None:
            f_txt = open(outf, "w")
        else:
            f_txt = None
    v = count_winds(all_empty[col].value_counts(), wind)
    vb = v[v > th]
    res = {}
    failed = []
    for i in vb.index:
        if (i * wind) < MIN_BASE:
            continue
        if f_txt is None:
            print("On %d-%d original reads" % (i * wind, (i + 1) * wind - 1))
        else:
            f_txt.write("On %d-%d original reads\n" % (i * wind, (i + 1) * wind - 1))
        inds = all_empty[(all_empty[col] >= i * wind) & (all_empty[col] < ((i + 1) * wind))].index
        lam, sc, teta_base = match_GP(inds, found, col, drop, f_txt)
        if sc < 1:
            res[i] = [lam, teta_base, len(inds)]
        else:
            failed.append(i)
    if len(failed) > 0:
        print("For: %s: failed" % str(failed))
    if len(res) == 0:
        return pandas.DataFrame()
    res = pandas.DataFrame(res).T
    if outf is not None:
        res.to_csv(outf.replace('.txt', '.csv'))
        try:
            plt.scatter(res.index, res[0].values)
            plt.ylim([-0.1, 1.1])
            plt.title("Lambda, each position as a window of length %d\n(%d out of shown range)" %
                      (wind, len(res[(res[0] < -0.1) | (res[0] > 1.1)])))
            plt.savefig(outf.replace('.txt', '_lam.png'))
            plt.close('all')
            plt.scatter(res.index, res[1].values)
            plt.title("Theta base (no div 1/(1-lam)), each position as a window of length %d" % wind)
            plt.savefig(outf.replace('.txt', '_tet_base.png'))
            plt.close('all')
        except:
            pass
    return res


def plot_hist(found, i, probs, lam, tet, path_hists):
    rng = list(range(int(max(found)) + 1))
    s = len(found)
    try:
        plt.hist(found, bins=rng, color='b')
        plt.plot(rng, s * probs[rng], color='r')
        plt.xlabel("Number of copies per oligo, from base %d" % i)
        plt.ylabel("Number of oligos (%d overall)" % len(found))
        plt.title("Hist of oligos from base %d vs GPD %g %g" % (i, lam, tet))
        plt.savefig(os.path.join(path_hists, "hist%d.jpg" % i))
        plt.close('all')
    except:
        pass


def calc_0(found, cur_empty, col, lam, tet_b, p_val, num_all, out_info_level, path_hists=None, f_txt=None):
    top = []
    if lam > MAX_LAM:
        if f_txt is None:
            print("For 0, lambda %g too high, capping at %g" % (lam, MAX_LAM))
        else:
            f_txt.write("For 0, lambda %g too high, capping at %g\n" % (lam, MAX_LAM))
        lam = MAX_LAM
    if lam < MIN_LAM:
        if f_txt is None:
            print("For 0, lambda %g too low, capping at %g" % (lam, MIN_LAM))
        else:
            f_txt.write("For 0, lambda %g too low, capping at %g\n" % (lam, MIN_LAM))
        lam = MIN_LAM
    tet = MIN_BASE * tet_b * (1 - lam)
    inds = list(cur_empty[cur_empty[col] == 0].index) + list(set(found.index).difference(set(cur_empty.index)))
    num_zeros = (num_all - len(cur_empty) + len(cur_empty[cur_empty[col] == 0].index))
    f = found.reindex(index=inds, columns=[col], fill_value=0)[col]
    if len(f.values) == 0:
        max_needed = 1
    else:
        max_needed = numpy.float64(max(f.values))
    log_cdf, probs = get_log10_cdf(lam, tet, max_needed, f_txt)
    if (out_info_level > 1) and (path_hists is not None):
        plot_hist(f.values.tolist() + [0] * (num_zeros - len(f)), 0, probs, lam, tet, path_hists)
    for ind in f.index:
        pos = int(f.loc[ind])
        if pos >= len(log_cdf):
            new_pos = len(log_cdf) - 1
            if f_txt is None:
                print("Using pos %d not %d, since log_cdf too short on %s" % (new_pos, pos, ind))
            else:
                f_txt.write("Using pos %d not %d, since log_cdf too short on %s\n" % (new_pos, pos, ind))
        else:
            new_pos = pos
        if (p_val == 0) or (log_cdf[new_pos] <= p_val):
            top.append([ind, 0, pos, -log_cdf[new_pos]])
    return top, len(f)


def calc_n(found, cur_empty, i, col, lam, tet_b, p_val, out_info_level, path_hists=None, f_txt=None):
    top = []
    if lam > MAX_LAM:
        if f_txt is None:
            print("For %d, lambda %g too high, capping at %g" % (i, lam, MAX_LAM))
        else:
            f_txt.write("For %d, lambda %g too high, capping at %g\n" % (i, lam, MAX_LAM))
        lam = MAX_LAM
    if lam < MIN_LAM:
        if f_txt is None:
            print("For %d, lambda %g too low, capping at %g" % (i, lam, MIN_LAM))
        else:
            f_txt.write("For %d, lambda %g too low, capping at %g\n" % (i, lam, MIN_LAM))
        lam = MIN_LAM
    tet = max(i, MIN_BASE) * tet_b * (1 - lam)
    inds = cur_empty[cur_empty[col] == i].index
    f = found.reindex(index=inds, columns=[col], fill_value=0)[col]
    if len(f.values) == 0:
        max_needed = 1
    else:
        max_needed = numpy.float64(max(f.values))
    log_cdf, probs = get_log10_cdf(lam, tet, max_needed, f_txt)
    if (out_info_level > 1) and (path_hists is not None):
        plot_hist(f.values, i, probs, lam, tet, path_hists)
    for ind in f.index:
        pos = int(f.loc[ind])
        if pos >= len(log_cdf):
            new_pos = len(log_cdf) - 1
            if f_txt is None:
                print("Using pos %d not %d, since log_cdf too short on %s" % (new_pos, pos, ind))
            else:
                f_txt.write("Using pos %d not %d, since log_cdf too short on %s\n" % (new_pos, pos, ind))
        else:
            new_pos = pos
        if (p_val == 0) or (log_cdf[new_pos] <= p_val):
            top.append([ind, i, pos, -log_cdf[new_pos]])
    return top, len(f)


def plot_regr(res, regr, path_hists, title=""):
    try:
        m = max(res.index)
        plt.scatter(res.index, res.values, color='b')
        plt.plot([0, m], [regr[0], regr[0] + m * regr[1]], color='r')
        plt.title("Regression of %s" % title)
        plt.savefig(os.path.join(path_hists, "regr_%s.jpg" % title))
        plt.close('all')
    except:
        pass


def fit_params(res, out_info_level, path_hists=None, wind=1, f_txt=None):
    # fit lambda with slope
    ransac = linear_model.RANSACRegressor()
    ransac.fit(res.index.values.reshape((-1, 1)), res[0].values)
    lam_p = [ransac.estimator_.intercept_, ransac.estimator_.coef_[0] / wind]

    regr_tet = linear_model.LinearRegression(fit_intercept=False)
    regr_tet.fit(res.index.values.reshape((-1, 1)), res[1].values)
    tet_b = regr_tet.coef_[0] / wind

    if f_txt is None:
        print("Working with lam=%g+%g*x tet=%g*x" % (lam_p[0], lam_p[1], tet_b))
    else:
        f_txt.write("Working with lam=%g+%g*x tet=%g*x\n" % (lam_p[0], lam_p[1], tet_b))

    if (out_info_level > 0) and (path_hists is not None):
        plot_regr(res[0], lam_p, path_hists, "lambda")
        plot_regr(res[1], [0, tet_b], path_hists, "theta")

    return lam_p, tet_b


def calc_top_p_values(lam_p, tet_b, cur_empty, found, col, out_info_level, num_all, path_hists=None, p_val=0.01,
                      f_txt=None):
    p_val = numpy.log10(p_val)

    if f_txt is None:
        print("Start (%d)" % len(found))
    else:
        f_txt.write("Start (%d)\n" % len(found))
    top, num_checked = calc_0(found, cur_empty, col, lam_p[0] + MIN_BASE * lam_p[1], tet_b, p_val,
                              num_all, out_info_level, path_hists, f_txt)
    if f_txt is None:
        print("Up to base %d, checked %d got %d. At" % (0, num_checked, len(top)), time.ctime())
    else:
        f_txt.write("Up to base %d, checked %d got %d. At " % (0, num_checked, len(top)) + time.ctime() + "\n")

    m = int(cur_empty[col].max()) + 1
    for i in range(1, m + 1):
        if len(cur_empty[cur_empty[col] == i]) == 0:
            continue
        t, n = calc_n(found, cur_empty, i, col, lam_p[0] + max(i, MIN_BASE) * lam_p[1], tet_b, p_val, out_info_level,
                      path_hists, f_txt)
        top += t
        num_checked += n
        if f_txt is None:
            print("Up to base %d, checked %d got %d" % (i, num_checked, len(top)))
        else:
            f_txt.write("Up to base %d, checked %d got %d\n" % (i, num_checked, len(top)))

    top = pandas.DataFrame(top, columns=['oligo', 'orig_cnt', 'final_cnt', '-log10_p'])
    top.sort_values('-log10_p', inplace=True, ascending=False)
    top.index = range(len(top))
    return top, num_checked


def get_log10_pmf(lam, tet, max_needed=10 ** 5, f_txt=None):
    log_probs = []
    x = Decimal(0)
    lam = Decimal(lam)
    tet = Decimal(tet)
    max_needed = Decimal(max_needed)
    e = Decimal.exp(Decimal(1))
    t = Decimal.log10(tet) + Decimal.log10(tet + x * lam) * (x - 1) + \
        Decimal.log10(e) * (-tet - x * lam)
    log_probs.append(t)
    x += 1
    factorial = Decimal.log10(x)
    while x <= max_needed:
        try:
            factorial += Decimal.log10(x)
            t = Decimal.log10(tet) + Decimal.log10(tet + x * lam) * (x - 1) + \
                Decimal.log10(e) * (-tet - x * lam) - factorial
        except:
            if f_txt is None:
                print("WTF! bad parameters %g %g failed on %d of %d" % (lam, tet, x, max_needed))
            else:
                f_txt.write("WTF! bad parameters %g %g failed on %d of %d\n" % (lam, tet, x, max_needed))
            return log_probs
        log_probs.append(t)
        x += 1
    return log_probs


def get_log10_cdf(lam, tet, max_needed=10 ** 5, f_txt=None):
    log_probs = get_log10_pmf(lam, tet, max_needed, f_txt)
    probs = 10 ** numpy.array(log_probs)
    cdf = numpy.cumsum(probs[::-1])[::-1] + max(1 - sum(probs), 0)
    if cdf[-1] <= 0:
        if f_txt is None:
            print("WTF. Negative CDFs", len(cdf), cdf[-5:])
        else:
            f_txt.write("WTF. Negative CDFs\n", len(cdf), cdf[-5:])
        cdf = cdf - cdf[-1] + Decimal(1) ** -(PRECISION - 1)
        if f_txt is None:
            print(cdf[-5:])
        else:
            f_txt.write(cdf[-5:] + "\n")
    log_cdf = numpy.log10(cdf)
    log_cdf = numpy.clip(log_cdf, -PRECISION, 0)
    return log_cdf, probs


def scatter_vs_non_found(found, base, tit_found, tit_base, col='all', num_all=244000, outf=None):
    re_base = found.sum() / base.sum()
    f = base * re_base
    f = f.merge(found, 'outer', left_index=True, right_index=True)
    f.fillna(0, inplace=True)
    x = list(f[col + "_x"].values)
    x += [0] * (num_all - len(x))
    y = list(f[col + "_y"].values)
    y += [0] * (num_all - len(y))
    try:
        plt.figure()
        plt.scatter(x, y)
        m = min(max(x), max(y))
        plt.plot([0, m], [0, m], color='r')
        plt.xlabel("Num reads in %s, normalized to %g" % (tit_base, found.sum()))
        plt.ylabel("Num reads in %s" % tit_found)
        plt.title("With antibodies vs. base")
        if outf is None:
            plt.show()
        else:
            plt.savefig(outf)
            plt.close('all')
    except:
        pass


def run_serum_vs_input_levels(input_levels_file, samp, cols, p_val, size_lib, out_path, out_info_level=0, pr_out=None):
    global MIN_BASE
    getcontext().prec = PRECISION
    col = 'good'

    if pr_out is None:
        print("base %s on samp %s, at" % (input_levels_file, samp), time.ctime())
    else:
        pr_out.write("base %s on samp %s, at " % (input_levels_file, samp) + time.ctime() + "\n")
    df_input_levels = read_file(input_levels_file, cols)
    if (df_input_levels[col].sum() / size_lib / 4) < MIN_BASE:
        MIN_BASE = max(int((df_input_levels.sum() / size_lib / 4)), 5)
        print("Warning!!! changed MIN_BASE to %d because of low input levels" % MIN_BASE)
        if pr_out is not None:
            pr_out.write("Warning!!! changed MIN_BASE to %d because of low input levels\n" % MIN_BASE)
    if len(df_input_levels) > size_lib:
        if pr_out is None:
            print("Warning. There should not be more mock variants (%d) then num_variantes (%d)" %
                  (len(df_input_levels), size_lib))
        else:
            pr_out.write("Warning. There should not be more mock variants (%d) then num_variantes (%d)\n" %
                         (len(df_input_levels), size_lib))
    if df_input_levels[col].sum() < (100 * size_lib):
        if pr_out is None:
            print("Warning. Only %g average reads per variant in mock" % (1. * df_input_levels[col].sum() / size_lib))
        else:
            pr_out.write("Warning. Only %g average reads per variant in mock\n" %
                         (1. * df_input_levels[col].sum() / size_lib))
    if df_input_levels[col].sum() < (20 * size_lib):
        print("Can't run on less then 20")
        return [None, None], None
    df_serum = read_file(samp, cols)
    if len(df_serum) > size_lib:
        if pr_out is None:
            print("Warning. There should not be more serum variants (%d) then num_variantes (%d)" % (len(df_serum),
                                                                                                     size_lib))
        else:
            pr_out.write("Warning. There should not be more serum variants (%d) then num_variantes (%d)\n" %
                         (len(df_serum), size_lib))
    input_levels_file = os.path.splitext(os.path.basename(input_levels_file))[0]
    samp = os.path.splitext(os.path.basename(samp))[0]
    if out_info_level > 0:
        path_hists = os.path.join(out_path, "hists_%s" % samp)
        if not os.path.exists(path_hists):
            os.makedirs(path_hists)
    else:
        path_hists = None

    if out_info_level > 2:
        scatter_vs_non_found(df_serum, df_input_levels, samp, input_levels_file, col=col, num_all=size_lib,
                             outf=os.path.join(out_path, "scatter_%s_vs_%s.jpg" % (input_levels_file, samp)))

    if out_info_level > 1:
        res = match_GPs(df_input_levels, df_serum, os.path.join(out_path, "lambdas_%s.txt" % samp), col,
                        drop=DROP, wind=1, th=MIN_CNT_BASE_CELL, f_txt=pr_out)
    else:
        res = match_GPs(df_input_levels, df_serum, None, col, drop=DROP, wind=1, th=MIN_CNT_BASE_CELL, f_txt=pr_out)
    pr_out.flush()
    if len(res) < MIM_VALS_TO_FIT:
        if pr_out is None:
            print("Only %d vals found. Can't estimate params" % len(res))
        else:
            pr_out.write("Only %d vals found. Can't estimate params\n" % len(res))
        return [None, None], None
    if pr_out is None:
        print("Fitting params for extrapulation from %d values" % len(res))
    else:
        pr_out.write("Fitting params for extrapulation from %d values\n" % len(res))

    lam_p, tet_b = fit_params(res, out_info_level, path_hists=path_hists, wind=1, f_txt=pr_out)
    top, num_checked = calc_top_p_values(lam_p, tet_b, df_input_levels, df_serum, col, out_info_level, p_val=p_val,
                                         num_all=size_lib, path_hists=path_hists, f_txt=pr_out)
    if pr_out is None:
        print("Found %d of %d to pass p_val of %g" % (len(top), num_checked, p_val))
    else:
        pr_out.write("Found %d of %d to pass p_val of %g\n" % (len(top), num_checked, p_val))
    top.to_csv(os.path.join(out_path, "top_samp%s.csv" % samp))
    pr_out.flush()
    print("End at", time.ctime())
    return lam_p, tet_b
