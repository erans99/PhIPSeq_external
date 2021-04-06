import math
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy
import pandas
import scipy

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
    grps = ['is_patho', 'is_probio', 'is_IgA', 'is_PNP', 'is_toxin']
    base_nums = []
    vals = []
    vals1 = []
    for grp in grps:
        vals.append(tmp[tmp[grp]][0].values.tolist())
        vals1.append((1 + tmp[tmp[grp]][0].values).tolist())
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


def check_commensal_abundances(olis, out_path):
    tmp = olis[olis.is_PNP]

    grps = ['high_abun', 'medium_abun', 'low_abun']
    vals = []
    vals1 = []
    base_nums = []
    m = 0
    for grp in grps:
        vals.append(tmp[tmp[grp] == 1][0].values.tolist())
        vals1.append((1 + tmp[tmp[grp] == 1][0].values).tolist())
        base_nums.append(len(vals[-1]))
        print(grp, base_nums[-1])
        m = max(m, max(vals[-1]))
    m = math.ceil(m / 100) * 10
    sc = ""
    for i in range(len(vals)):
        for j in range(i):
            scs = [scipy.stats.ks_2samp(vals[i], vals[j], 'less')[1],
                   scipy.stats.ks_2samp(vals[i], vals[j], 'greater')[1]]
            sc += "[%.2g, %.2g] " % (scs[0], scs[1])
            print("%s vs %s: %g %g" % (grps[i], grps[j], scs[0], scs[1]))

    legend = ["%s (of %d)" % (grps[i], base_nums[i]) for i in range(len(grps))]
    my_hist(vals, legend, range(0, m * 10 + 9, m), "Histogram of number of samples each oligo appears in\n(%s)" % sc,
            os.path.join(out_path, "%s_EMabundance_hist_num_samples_per_oli.png" % lib))

    my_log_hist(vals1, legend, range(1, m * 10 + 9, m),
                "Histogram of number of samples each oligo appears in\n(%s)" % sc,
                os.path.join(out_path, "%s_EMabundance_hist_num_samples_per_oli_log.png" % lib))
    my_log_perc_graph(vals1, legend, range(1, m * 10 + 9, m),
                      "Histogram of number of samples each oligo appears in\n(%s)" % sc,
                      os.path.join(out_path, "%s_EMabundance_hist_num_samples_per_oli_perclog.png" % lib))
    my_log_cum_perc_graph(vals1, legend, range(1, m * 10 + 9, m),
                          "Histogram of number of samples each oligo appears in\n(%s)" % sc,
                          os.path.join(out_path, "%s_EMabundance_hist_num_samples_per_oli_perccumlog.png" % lib))


def check_function(olis, out_path):
    tmp = olis[olis.is_PNP]

    grps = ['is_bac_membrame', 'is_bac_flagella', 'is_bac_secrete', 'control']
    vals = []
    base_nums = []
    m = 0
    for grp in grps:
        if grp != 'control':
            vals.append(tmp[tmp[grp]][0].values.tolist())
        else:
            vals.append(tmp[(tmp['EM_high_abun'] | tmp['EM_medium_abun'] | tmp['EM_low_abun']) &
                            (tmp['is_other'])][0].values.tolist())
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
    plt.savefig(os.path.join(out_path, "%s_func_part_hist_num_samples_per_oli.png" % lib))
    plt.close("all")


def check_topgraph(olis, out_path):
    tmp = olis[olis.is_PNP]

    grps = ['full_topgraph', 'is_topgraph']
    vals = []
    base_nums = []
    m = 0
    for grp in grps:
        if grp != 'control':
            vals.append(tmp[tmp[grp] == 1][0].values.tolist())
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
    plt.savefig(os.path.join(out_path, "%s_topgraph_hist_num_samples_per_oli.png" % lib))
    plt.close("all")

    pname_full = list(tmp[tmp['full_topgraph'] == 1].prot_name.value_counts().index)
    df_cmp = tmp[tmp['is_topgraph'] & numpy.isin(tmp['prot_name'], pname_full)]
    vals = []
    vals.append(tmp[tmp['full_topgraph'] == 1][0].values.tolist())
    vals.append(df_cmp[0].values.tolist())
    base_nums = [len(vals[0]), len(vals[1])]
    m = math.ceil(max(max(vals[0]), max(vals[1])) / 100) * 10
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
    plt.title("Histogram of number of samples each oligo appears in\n(only proteins which appear in both)(%s)" % sc)
    plt.yticks(10 ** numpy.arange(6), 10 ** numpy.arange(6))
    plt.ylim([0.9, 10 ** 6])
    plt.xlabel("Number of individuals")
    plt.ylabel("Log number of significantly enriched epitopes")
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "%s_topgraph_partial_hist_num_samples_per_oli.png" % lib))
    plt.close("all")


def check_strain_groups(olis, out_path):
    tmp = olis[olis.is_PNP | olis.is_nonPNP_strains]

    grps = ['is_MPA', 'is_IgA', 'is_patho', 'is_probio']
    vals = []
    base_nums = []
    m = 0
    for grp in grps:
        is_grp = tmp[tmp[grp]]
        for g in grps:
            if g != grp:
                is_grp = is_grp[~is_grp[g]]
        vals.append(is_grp[0].values.tolist())
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


def check_secreted(olis, out_path, with_tox=False):
    tmp = olis

    grps = ['is_bac_toxin', 'is_bac_secrete']
    if with_tox:
        grps.append('is_toxin')
    vals = []
    base_nums = []
    m = 0
    for grp in grps:
        is_grp = tmp[tmp[grp]]
        for g in grps:
            if g != grp:
                is_grp = is_grp[~is_grp[g]]
        vals.append(is_grp[0].values.tolist())
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
    if with_tox:
        plt.savefig(os.path.join(out_path, "%s_secrete_vs_toxins_hist_num_samples_per_oli.png" % lib))
    else:
        plt.savefig(os.path.join(out_path, "%s_secrete_vs_toxin_hist_num_samples_per_oli.png" % lib))
    plt.close("all")


def before_and_after_pie(olis, out_path, min_p, num_all):
    min_val = min_p * num_all
    # 'is_pos_cntrl', 'is_neg_cntrl', 'is_rand_cntrl'
    grps = ['is_patho', 'is_probio', 'is_IgA', 'is_PNP', 'is_toxin']
    vals = [[], []]
    for i, grp in enumerate(grps):
        is_grp = olis[olis[grp]]
        cnt = len(is_grp)
        for g in grps:
            if g != grp:
                is_grp = is_grp[~is_grp[g]]
        vals[0].append(len(is_grp))
        vals[1].append(len(is_grp[is_grp[0] > min_val]))
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


def check_strains(olis, out_path, in_strains, num_all, with_over=False):
    strains = pandas.read_csv(in_strains, index_col=0, encoding='latin-1')
    grps = ['IgA', 'probi', 'patho', 'PNP']
    names = [[], []]
    for st in strains.index:
        grp = strains.loc[st].group_1.split()[0][:5]
        if grp in grps:
            names[0].append(strains.loc[st].NCBI_name)
            names[1].append(grp)
        else:
            print("What is %s %s" % (grp, strains.loc[st].group_1))
    vals = {}
    for v in olis.bac_src.value_counts().index:
        if v in names[0]:
            if v in vals.keys():
                vals[v] += list(olis[olis.bac_src == v][0].values)
            else:
                vals[v] = list(olis[olis.bac_src == v][0].values)
        elif with_over:
            try:
                ps = v.split(" & ")
            except:
                continue
            for p in ps:
                if p in names[0]:
                    if p in vals.keys():
                        vals[p] += list(olis[olis.bac_src == v][0].values)
                    else:
                        vals[p] = list(olis[olis.bac_src == v][0].values)
    vals = [vals[x] for x in names[0]]

    res = {}
    base_nums = []
    cnt = 0
    for i, n in enumerate(names[0]):
        base_nums.append(len(vals[i]))
        res[names[0][i]] = [names[1][i], base_nums[i]]
        cnt += 1
    cols = ['grp', 'num_base']

    min_ps = [0.01, 0.03, 0.05, 0.1, 0.2, 0.5, 0.9]
    for min_p in min_ps:
        # stats = {}
        min_val = min_p * num_all
        base_nums = []
        for i, n in enumerate(names[0]):
            base_nums.append(sum(numpy.array(vals[i]) > min_val))
            res[n] += [base_nums[i], (100 * base_nums[i] / res[n][1])]
        cols += ['num_over_%g%%' % (100 * min_p), 'perc_over_%g%%' % (100 * min_p)]
        # if names[1][i] in stats.keys():
        #     stats[names[1][i]].append((100 * base_nums[1][i] / base_nums[0][i]))
        # else:
        #     stats[names[1][i]] = [(100 * base_nums[1][i] / base_nums[0][i])]

    res = pandas.DataFrame(res, index=cols).T
    res.sort_values('perc_over_5%', ascending=False, inplace=True)
    if with_over:
        res.to_csv(os.path.join(out_path, "%s_strains_overlap_num_olis_above_th.csv" % lib))
    else:
        res.to_csv(os.path.join(out_path, "%s_strains_no_over_num_olis_above_th.csv" % lib))
    res.sort_values("grp", inplace=True)
    colors = {'IgA': "r", "PNP": "b", "patho": "g", "probi": "y"}
    for perc in [3, 5, 10, 20]:
        plt.bar(range(len(res)), res['perc_over_%d%%' % perc].values, color=[colors[x] for x in res['grp'].values])
        plt.xticks(range(len(res)), res.index, rotation='vertical')
        plt.title("Percentage of strain appearing in above %d%% of individuals" % perc)
        plt.tight_layout()
        if with_over:
            plt.savefig(os.path.join(out_path, "%s_strains_overlap_above_%d.png" % (lib, perc)))
        else:
            plt.savefig(os.path.join(out_path, "%s_strains_no_overlap_above_%d.png" % (lib, perc)))
        plt.close("all")


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
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS)))
        metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s_above%d.pkl" % (lib, MIN_OLIS)))

    print(exist_df.shape)
    print(metadata.columns)
    print(metadata.shape)
    print(time.ctime())

    if not os.path.isfile(os.path.join(cache_path, "%s_protein_exist.pkl" % lib)):
        print("need to make protein exits. exiting")
        sys.exit(1)
    else:
        proteins_exist = pandas.read_pickle(os.path.join(cache_path, "%s_protein_exist.pkl" % lib))

    exist_df[exist_df < 0] = 0  # this is because -1 is "not_scored", and not a value.
    num_olis_per_sample = exist_df.sum(1)
    num_samples_per_oli = exist_df.sum(0)
    print(num_olis_per_sample)
    num_olis_per_sample.to_csv(os.path.join(base_path, "num_%s_olis_per_sample.csv" % lib))

    plt.boxplot(num_olis_per_sample.values)
    plt.title("Boxplot of number of oligos which passed threshold p_value")
    plt.ylabel("num_oligos")
    plt.savefig(os.path.join(out_path, "%s_box_num_passed.png" % lib))
    plt.close("all")

    print("For library %s" % lib)
    for i in num_olis_per_sample[num_olis_per_sample < 100].sort_values().index:
        print(i, num_olis_per_sample.loc[i])
        print("num_read", metadata.loc[i[0], 'num_reads'])
        print('comment', metadata.loc[i[0], 'comment'])
        print('params', metadata.loc[i[0], 'params'])
        print('num_passed', metadata.loc[i[0], 'num_passed'])
        print()

    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent_final_allsrc.pkl"))
    cols = list(df_info.columns)
    for c in ['end0_len15', 'hash0_len15', 'end1_len15', 'hash1_len15', 'end2_len14',
              'hash2_len14', 'end3_len15', 'hash3_len15', 'end4_len16', 'hash4_len16']:
        cols.pop(cols.index(c))
    df_info = df_info[cols]

    df_prot_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent_prot.pkl"))

    tmp = num_samples_per_oli.copy()
    tmp = tmp.reset_index()
    tmp.index = [int(x.split("_")[1]) for x in tmp.order]
    tmp = tmp[[0]]
    tmp = df_info.merge(tmp, "outer", left_index=True, right_index=True)
    tmp.fillna(0, inplace=True)

    num_samples_per_oli.index = num_samples_per_oli.index.get_level_values(0)
    inds = num_samples_per_oli.index.intersection(["agilent_%d" % x for x in
                                                   df_info[~df_info.is_IEDB_or_cntrl].index])
    print("Checking major groups")
    check_groups(tmp, out_path, metadata, len(df_info[~df_info.is_IEDB_or_cntrl]), len(exist_df),
                 num_samples_per_oli.loc[inds])
    num_samples_per_prot = proteins_exist.sum(0)
    plt.hist(num_samples_per_prot.values.tolist(), log=True, bins=range(0, len(metadata) + 24, 25))
    plt.xticks(range(0, len(metadata), 100), rotation='vertical')
    plt.title("Histogram of number of samples each protein appears in")
    plt.ylim([1, plt.ylim()[1]])
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "%s_hist_num_samples_per_prot.png" % lib))
    plt.close("all")

    check_commensal_abundances(tmp, out_path)
    check_function(tmp, out_path)
    check_topgraph(tmp, out_path)
    check_strain_groups(tmp, out_path)
    res = before_and_after_pie(tmp, out_path, 0.05, len(exist_df))
    check_secreted(tmp, out_path)
    check_secreted(tmp, out_path, True)
    check_strains(tmp, out_path, os.path.join(base_path, "match_strains_to_Segata", "strains_summ.csv"),
                  len(exist_df))
    check_strains(tmp, out_path, os.path.join(base_path, "match_strains_to_Segata", "strains_summ.csv"),
                  len(exist_df), True)

    res = []
    mins = [0, 0.01, 0.05, 0.1, 0.5, 0.9]
    for p in mins:
        res.append(before_and_after_pie(tmp, None, p, len(exist_df)))
    pandas.DataFrame(res, index=mins, columns=['is_patho', 'is_probio', 'is_IgA', 'is_PNP', 'is_toxin']).to_csv(
        os.path.join(out_path, "src_pass_by_prob.csv"))
    print()
