import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy
import pandas
import scipy
import scipy.stats

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

COLS = ['g', 'b', 'r', 'y', 'c', 'm']
lib = "agilent"

ALL = False
MIN_OLIS = 200


def my_hist(vals, grps, bins, title, outfile):
    plt.hist(vals, stacked=False, log=True, bins=bins, color=COLS[:len(grps)], label=grps)
    plt.legend(prop={'size': 10})
    plt.xticks(range(0, len(metadata), 100), rotation='vertical')
    plt.title(title)
    ym = plt.ylim()[1]
    yr = [1]
    while yr[-1] < ym:
        yr.append(yr[-1] * 10)
    plt.yticks(yr, yr)
    plt.ylim([0.9, yr[-1]])
    plt.xlabel("Number of individuals")
    plt.ylabel("Number of significantly enriched epitopes (log scale)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close("all")


def my_log_hist(vals, grps, bins, title, outfile):
    logbins = numpy.logspace(numpy.log10(bins[0]), numpy.log10(bins[-1]), len(bins))
    plt.hist(vals, stacked=False, log=True, bins=logbins, color=COLS[:len(grps)], label=grps)
    plt.xscale('log')

    plt.legend(prop={'size': 10})
    xm = plt.xlim()
    xr = [1]
    while xr[-1] < xm[1]:
        xr.append(1 + 10 ** len(xr))
    xr = xr[:-1]
    plt.xticks(xr, list(numpy.array(xr) - 1), rotation='vertical')
    plt.xlim(xm)
    plt.title(title)
    ym = plt.ylim()[1]
    yr = [1]
    while yr[-1] < ym:
        yr.append(yr[-1] * 10)
    plt.yticks(yr, yr)
    plt.ylim([0.9, yr[-1]])
    plt.xlabel("Number of individuals (log scale)")
    plt.ylabel("Number of significantly enriched epitopes (log scale)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close("all")


def my_log_perc_graph(vals, grps, bins, title, outfile):
    logbins = numpy.logspace(numpy.log10(bins[0]), numpy.log10(bins[-1]), len(bins))
    plt.hist(vals, stacked=False, log=True, bins=logbins, color=COLS[:len(grps)], label=grps, density=True)
    plt.xscale('log')

    plt.legend(prop={'size': 10})
    xm = plt.xlim()
    xr = [1]
    while xr[-1] < xm[1]:
        xr.append(1 + 10 ** len(xr))
    xr = xr[:-1]
    plt.xticks(xr, list(numpy.array(xr) - 1), rotation='vertical')
    plt.xlim(xm)
    plt.title(title)
    ym = plt.ylim()
    st = math.ceil(numpy.log10(ym[0]))
    yr = []
    yrp = []
    while st <= 0:
        yr.append(10 ** st)
        yrp.append("%g%%" % (100 * 10 ** st))
        st += 1
    plt.yticks(yr, yrp)
    plt.ylim(ym)
    plt.xlabel("Number of individuals (log scale)")
    plt.ylabel("Percent of significantly enriched\nepitopes (log scale)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close("all")


def my_log_cum_perc_graph(vals, grps, bins, title, outfile):
    logbins = numpy.logspace(numpy.log10(bins[0]), numpy.log10(bins[-1]), len(bins))
    plt.hist(vals, stacked=False, log=True, bins=logbins, color=COLS[:len(grps)], label=grps, density=True,
             cumulative=-1)
    plt.xscale('log')

    plt.legend(prop={'size': 10})
    xm = plt.xlim()
    xr = [1]
    while xr[-1] < xm[1]:
        xr.append(1 + 10 ** len(xr))
    xr = xr[:-1]
    plt.xticks(xr, list(numpy.array(xr) - 1), rotation='vertical')
    plt.xlim(xm)
    plt.title(title)
    ym = plt.ylim()
    st = math.ceil(numpy.log10(ym[0]))
    yr = []
    yrp = []
    while st <= 0:
        yr.append(10 ** st)
        yrp.append("%g%%" % (100 * 10 ** st))
        st += 1
    plt.yticks(yr, yrp)
    plt.ylim(ym)
    plt.xlabel("Number of individuals (log scale)")
    plt.ylabel("Reverse cummulative percent of significantly\nenriched epitopes (log scale)")
    plt.tight_layout()
    plt.savefig(outfile)
    plt.close("all")


def check_groups(olis, out_path, metadata, tot_olis, tot_cohort, num_samples_per_oli):
    tmp = olis[~olis.is_IEDB_or_cntrl]
    grps = ['is_patho_strain', 'is_probio_strain', 'is_IgA_coated_strain', 'is_gut_microbiome', 'is_VFDB']
    base_nums = []
    vals = []
    vals1 = []
    for grp in grps:
        vals.append(tmp[tmp[grp]]['num_samples_per_oli'].values.tolist())
        vals1.append((1 + tmp[tmp[grp]]['num_samples_per_oli'].values).tolist())
        base_nums.append(len(vals[-1]))
        tmp = tmp.drop(tmp[tmp[grp]].index)
        print(grp, len(vals[-1]), len(tmp))

    legend = ["%s (of %d)" % (grps[i], base_nums[i]) for i in range(len(grps))]
    my_hist(vals, legend, range(0, len(metadata) + 99, 100), "Histogram of number of samples each oligo appears in",
            os.path.join(out_path, "%s_abs_hist_num_samples_per_oli.png" % lib))
    my_log_hist(vals1, legend, range(1, len(metadata) + 99, 100),
                "Histogram of number of samples each oligo appears in",
                os.path.join(out_path, "%s_abs_hist_num_samples_per_oli_log.png" % lib))
    my_log_perc_graph(vals1, legend, range(1, len(metadata) + 99, 100),
                      "Histogram of number of samples each oligo appears in",
                      os.path.join(out_path, "%s_abs_hist_num_samples_per_oli_perclog.png" % lib))
    my_log_cum_perc_graph(vals1, legend, range(1, len(metadata) + 99, 100),
                          "Histogram of number of samples each oligo appears in",
                          os.path.join(out_path, "%s_abs_hist_num_samples_per_oli_perccumlog.png" % lib))

    plt.hist(num_samples_per_oli.values.tolist() + [0] * (tot_olis - len(num_samples_per_oli)), log=True,
             bins=range(0, len(metadata) + 24, 25))
    plt.xticks(range(0, len(metadata), 100), rotation='vertical')
    plt.title("Histogram of number of samples each oligo appears in")
    plt.yticks(10 ** numpy.arange(6), 10 ** numpy.arange(6))
    plt.ylim([1, 10 ** 6])
    plt.xlabel("Number of individuals")
    plt.ylabel("Log number of significantly enriched epitopes")
    plt.tight_layout()
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        plt.vlines(th * tot_cohort, plt.ylim()[0], plt.ylim()[1] / 2, color='g', linestyles="dashed")
        plt.text(th * tot_cohort + 10, 5000, "%d olis in > %g%% of cohort" % (
            len(num_samples_per_oli[num_samples_per_oli > (th * tot_cohort)]), th * 100), rotation=90,
                 verticalalignment='center', color='g')
    plt.savefig(os.path.join(out_path, "%s_hist_num_samples_per_oli.png" % lib))
    plt.close("all")
    bins = range(1, len(metadata) + 24, 25)
    logbins = numpy.logspace(numpy.log10(bins[0]), numpy.log10(bins[-1]), len(bins))
    plt.hist((1 + num_samples_per_oli.values).tolist() + [1] * (tot_olis - len(num_samples_per_oli)), log=True,
             bins=logbins)
    plt.xscale('log')
    plt.title("Histogram of number of samples each bacterial oligo appears in")
    plt.xlabel("Number of individuals")
    plt.ylabel("Log number of significantly enriched epitopes")
    plt.tight_layout()
    for th in [0.9, 0.7, 0.5, 0.3, 0.1]:
        plt.vlines(th * tot_cohort, plt.ylim()[0], plt.ylim()[1] / 2, color='g', linestyles="dashed")
        plt.text(th * tot_cohort + 10, 5000, "%d olis in > %g%% of cohort" % (
            len(num_samples_per_oli[num_samples_per_oli > (th * tot_cohort)]), th * 100), rotation=90,
                 verticalalignment='center', color='g')
    plt.savefig(os.path.join(out_path, "%s_hist_num_samples_per_oli_log.png" % lib))
    plt.close("all")


def check_strain_groups(olis, out_path):
    tmp = olis[~olis.is_IEDB_or_cntrl]

    grps = ['is_gut_microbiome', 'is_IgA_coated_strain', 'is_patho_strain', 'is_probio_strain']
    vals = []
    base_nums = []
    m = 0
    for grp in grps:
        is_grp = tmp[tmp[grp]]
        for g in grps:
            if g != grp:
                is_grp = is_grp[~is_grp[g]]
        vals.append(is_grp['num_samples_per_oli'].values.tolist())
        base_nums.append(len(vals[-1]))
        print(grp, base_nums[-1])
        m = max(m, max(vals[-1]))
    m = math.ceil(m / 100) * 10
    plt.hist(vals, stacked=False, log=True, bins=range(0, m * 10 + 9, m), color=COLS[:len(grps)],
             label=["%s (of %d)" % (grps[i], base_nums[i]) for i in range(len(grps))])
    plt.legend(prop={'size': 10})
    plt.xticks(range(0, m * 10 + 9, m), rotation='vertical')
    sc = ""
    for i in range(len(vals)):
        for j in range(i):
            scs = [scipy.stats.ks_2samp(vals[i], vals[j], 'less')[1],
                   scipy.stats.ks_2samp(vals[i], vals[j], 'greater')[1]]
            sc += "[%.2g, %.2g] " % (scs[0], scs[1])
            print("%s vs %s: %g %g" % (grps[i], grps[j], scs[0], scs[1]))
    plt.title("Histogram of number of samples each oligo appears in\n(%s)" % sc)
    plt.yticks(10 ** numpy.arange(6), 10 ** numpy.arange(6))
    plt.ylim([0.9, 10 ** 6])
    plt.xlabel("Number of individuals")
    plt.ylabel("Log number of significantly enriched epitopes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "%s_strain_grps_hist_num_samples_per_oli.png" % lib))
    plt.close("all")


def before_and_after_pie(olis, out_path, min_p, num_all):
    min_val = min_p * num_all
    # 'is_pos_cntrl', 'is_neg_cntrl', 'is_rand_cntrl'
    grps = ['is_patho_strain', 'is_probio_strain', 'is_IgA_coated_strain', 'is_gut_microbiome', 'is_VFDB']
    vals = [[], []]
    for i, grp in enumerate(grps):
        is_grp = olis[olis[grp]]
        cnt = len(is_grp)
        for g in grps:
            if g != grp:
                is_grp = is_grp[~is_grp[g]]
        vals[0].append(len(is_grp))
        vals[1].append(len(is_grp[is_grp['num_samples_per_oli'] > min_val]))
        if vals[0][-1] != cnt:
            print(grp, vals[0][-1], vals[1][-1], "(%d)" % cnt)
        else:
            print(grp, vals[0][-1], vals[1][-1])
    print(sum(vals[0]), sum(vals[1]))
    if min_p == 0:
        return vals[0]

    if out_path is not None:
        ax = plt.subplot2grid((1, 5), (0, 0), colspan=2)
        plt.pie(vals[0], autopct='%d%%')
        plt.legend(grps, bbox_to_anchor=(1, 0, 0.2, 1))
        plt.title('All oligos')

        ax = plt.subplot2grid((1, 5), (0, 3), colspan=2)
        plt.pie(vals[1], autopct='%d%%')
        plt.title('Shown in > %d%%' % (100 * min_p))

        plt.savefig(os.path.join(out_path, "%s_pie_before_and_after_%g.png" % (lib, min_p)))
        plt.close("all")
    return vals[1]


if __name__ == '__main__':
    cache_path = os.path.join(base_path, "Cache")
    if ALL:
        out_path = os.path.join(base_path, "descript", "All")
    else:
        out_path = os.path.join(base_path, "descript", "Above_%d" % MIN_OLIS)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    print(time.ctime())
    # data = pandas.read_pickle(os.path.join(cache_path, "fold_%s.pkl" % lib)).T
    if ALL:
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s.pkl" % lib)).T
        metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s.pkl" % lib)).T
    else:
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s_above%d.pkl" % (lib, MIN_OLIS))).T

    print(exist_df.shape)
    print(metadata.columns)
    print(metadata.shape)
    print(time.ctime())

    exist_df[exist_df < 0] = 0  # this is because -1 is "not_scored", and not a value.
    num_olis_per_sample = exist_df.sum(1)
    num_samples_per_oli = exist_df.sum(0)
    print(num_olis_per_sample)
    num_olis_per_sample.to_csv(os.path.join(out_path, "num_%s_olis_per_sample.csv" % lib))

    plt.boxplot(num_olis_per_sample.values)
    plt.title("Boxplot of number of oligos which passed threshold p_value")
    plt.ylabel("num_oligos")
    plt.savefig(os.path.join(out_path, "%s_box_num_passed.png" % lib))
    plt.close("all")

    print("For library %s" % lib)
    for i in num_olis_per_sample[num_olis_per_sample < 100].sort_values().index:
        print(i, num_olis_per_sample.loc[i])
        print(metadata.loc[i])
        print()

    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_%s.pkl" % lib))
    df_info['is_IEDB_or_cntrl'] = df_info.is_IEDB | df_info.is_pos_cntrl | df_info.is_neg_cntrl | df_info.is_rand_cntrl

    tmp = num_samples_per_oli.copy()
    tmp.name = 'num_samples_per_oli'
    tmp = df_info.merge(tmp, "outer", left_index=True, right_index=True)
    tmp.fillna(0, inplace=True)

    inds = num_samples_per_oli.index.intersection(df_info[~df_info.is_IEDB_or_cntrl].index)
    print("Checking major groups")
    check_groups(tmp, out_path, metadata, len(df_info[~df_info.is_IEDB_or_cntrl]), len(exist_df),
                 num_samples_per_oli.loc[inds])

    check_strain_groups(tmp, out_path)
    res = before_and_after_pie(tmp, out_path, 0.05, len(exist_df))
    res = []
    mins = [0, 0.01, 0.05, 0.1, 0.5, 0.9]
    for p in mins:
        res.append(before_and_after_pie(tmp, None, p, len(exist_df)))
    pandas.DataFrame(res, index=mins, columns=['is_patho', 'is_probio', 'is_IgA', 'is_PNP', 'is_toxin']).to_csv(
        os.path.join(out_path, "src_pass_by_prob.csv"))
    print()
