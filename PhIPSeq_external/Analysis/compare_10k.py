import os

import matplotlib.pyplot as plt
import numpy
import pandas

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

params = {'colsample_bytree': 0.1, 'max_depth': 4, 'learning_rate': 0.001, 'n_estimators': 6000, 'subsample': 0.6,
          'min_child_weight': 0.01}

cache_path = os.path.join(base_path, "Cache")

MIN_OLIS = 200
PLOT_SINGLES = False
ALL = False


def plot_dup(outpath, df, p, name1, date1, name2, date2, is_log=False, force=False):
    if is_log:
        fname = os.path.join(outpath, "plot_log_%s_%s.png")
    else:
        fname = os.path.join(outpath, "plot_%s_%s.png")
    if force or (not os.path.exists(os.path.join(outpath, "%s_%s.png" % (name1, name2)))):
        plt.figure()
        plt.scatter(df[name1], df[name2])
        plt.xlabel(str(name1))
        plt.ylabel(str(name2))
        plt.title("Repeating individual (corr P=%g)\nat %s and %s" % (p, date1, date2))
        plt.savefig(fname % (name1, name2))
        plt.close("all")
    return


def comp_R2(act, pred):
    inds = numpy.where(~numpy.isnan(act))[0]
    m = sum(act[inds]) / len(inds)
    SSres = sum((act[inds] - pred[inds]) ** 2)
    SStot = sum((act[inds] - m) ** 2)
    return 1 - SSres / SStot


def run_on_inds(outpath, k, inds, base, repeat10k, run_log=False, run_spearman=False):
    if ALL:
        meta_df = pandas.read_pickle(os.path.join(cache_path, "meta_matched_%s.pkl" % lib)).T
        fold_df = pandas.read_pickle(os.path.join(cache_path, "fold_matched_%s.pkl" % lib)).T
    else:
        meta_df = pandas.read_pickle(os.path.join(cache_path, "meta_matched_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        fold_df = pandas.read_pickle(os.path.join(cache_path, "fold_matched_%s_above%d.pkl" % (lib, MIN_OLIS))).T

    fold_df.fillna(1, inplace=True)
    fold_df[fold_df < 1] = 1
    fold_df.columns = fold_df.columns.get_level_values(0)
    logfold_df = numpy.log(fold_df)

    res = {}
    for i, b in enumerate(base):
        if (i % 10) == 0:
            print("At %d or %d" % (i, len(base)))
        tmp = {}
        for j, r in enumerate(repeat10k):
            if run_log:
                comp = logfold_df.loc[[b, r], inds].dropna(1, 'all').T
            else:
                comp = fold_df.loc[[b, r], inds].dropna(1, 'all').T
            comp.fillna(0, inplace=True)
            if run_spearman:
                tmp[r] = comp.corr('spearman').loc[b, r]
            else:
                tmp[r] = comp.corr('pearson').loc[b, r]
            if PLOT_SINGLES and (k == 'All') and (i == j):
                plot_dup(outpath, comp, tmp[r], b, meta_df.loc[b].Date, r, meta_df.loc[r].Date, run_log)
            res[b] = tmp

    res = pandas.DataFrame(res).T
    res = res.loc[base, repeat10k]
    if run_log:
        res.to_csv(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("log_fold", k)))
    elif run_spearman:
        res.to_csv(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("spearman", k)))
    else:
        res.to_csv(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("fold", k)))


if __name__ == "__main__":
    lib = "agilent"

    if ALL:
        meta_df = pandas.read_pickle(os.path.join(cache_path, "meta_matched_%s.pkl" % lib)).T
        fold_df = pandas.read_pickle(os.path.join(cache_path, "fold_matched_%s.pkl" % lib)).T
    else:
        meta_df = pandas.read_pickle(os.path.join(cache_path, "meta_matched_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        fold_df = pandas.read_pickle(os.path.join(cache_path, "fold_matched_%s_above%d.pkl" % (lib, MIN_OLIS))).T

    fold_df.fillna(1, inplace=True)
    fold_df[fold_df < 1] = 1
    fold_df.columns = fold_df.columns.get_level_values(0)

    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_%s.pkl" % lib))
    df_info['is_IEDB_or_cntrl'] = df_info.is_IEDB | df_info.is_pos_cntrl | df_info.is_neg_cntrl | df_info.is_rand_cntrl

    inds = {}
    inds['All'] = fold_df.columns.intersection(df_info[~df_info.is_IEDB_or_cntrl].index)
    inds['VFDB'] = fold_df.columns.intersection(df_info[df_info.is_VFDB].index)
    inds['microbiome'] = fold_df.columns.intersection(df_info[df_info.is_gut_microbiome].index)

    base = meta_df.index[::2]
    repeat10k = meta_df.index[1::2]
    if lib is None:
        outpath = os.path.join(base_path, "dup_10k")
    else:
        outpath = os.path.join(base_path, "dup_10k_%s" % lib)
    if ALL:
        outpath = os.path.join(outpath, "All")
    else:
        outpath = os.path.join(outpath, "Above_%d" % MIN_OLIS)

    if not os.path.exists(outpath):
        os.makedirs(outpath)

    del fold_df


    for k in inds.keys():
        if not os.path.exists(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("fold", k))):
            run_on_inds(outpath, k, inds[k], base, repeat10k, False, False)
        if not os.path.exists(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("spearman", k))):
            run_on_inds(outpath, k, inds[k], base, repeat10k, False, True)
        if not os.path.exists(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("log_fold", k))):
            run_on_inds(outpath, k, inds[k], base, repeat10k, True, False)
    print("Done")

    res = {}
    for n in ['spearman', "log_fold", "fold"]:
        res[n] = {}

    for k in inds.keys():
        for n in ['spearman', "log_fold", "fold"]:
            print("Working on %s on %s" % (n, k))
            r = pandas.read_csv(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % (n, k)), index_col=0)
            plt.figure()
            plt.imshow(r.values, cmap='plasma')
            plt.xlim(0, len(r))
            plt.ylim(0, len(r))

            plt.xticks(range(0, len(r), 10),
                       [meta_df.loc[x].RegistrationCode + " " + meta_df.loc[x].Date[:4] for x in base[::10]],
                       rotation='vertical')
            plt.yticks(range(0, len(r), 10),
                       [meta_df.loc[x].old_RegistrationCode + " " + meta_df.loc[x].Date[:4] for x in repeat10k[::10]],
                       rotation='horizontal')
            cbar = plt.colorbar()
            if n == 'spearman':
                cbar.ax.set_ylabel("spearman correlation", rotation=-90, va="bottom")
            else:
                cbar.ax.set_ylabel("pearson correlation", rotation=-90, va="bottom")
            plt.title("%s correlation on %s of old and new samples\nDiagonal is same person after 5 years" % (n, k))
            plt.tight_layout()
            plt.savefig(os.path.join(outpath, "%s_on_%s_corr_5years_all.png" % (n, k)))
            plt.close("all")
            vals = [[], []]
            for i in range(len(r)):
                v = r.iloc[i].values.tolist()
                vals[0].append(v.pop(i))
                vals[1] += v
                if r.iloc[i, i] != r.iloc[i].max():
                    print("Matched not best for:", i, r.iloc[i].name, r.T.iloc[i].name)
            res[n][(k, 'between')] = [numpy.array(vals[1]).mean(), numpy.array(vals[1]).std()]
            res[n][(k, 'within')] = [numpy.array(vals[0]).mean(), numpy.array(vals[0]).std()]

            fig, ax1 = plt.subplots()

            color = 'tab:red'
            if n == 'spearman':
                ax1.set_xlabel('spearman corr')
            else:
                ax1.set_xlabel('pearson corr')
            ax1.set_ylabel('between', color=color)
            ax1.hist(vals[1], label="different individual", bins=100, color=color, alpha=0.5)
            ax1.tick_params(axis='y', labelcolor=color)

            ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
            color = 'tab:blue'
            ax2.set_ylabel('within', color=color)  # we already handled the x-label with ax1
            ax2.hist(vals[0], label="same individual", bins=30, color=color, alpha=0.5)
            ax2.tick_params(axis='y', labelcolor=color)

            fig.tight_layout()  # otherwise the right y-label is slightly clipped
            if n == 'spearman':
                plt.title("Histogram of %s correlation on %s within individual vs between" % (n, k))
            else:
                plt.title("Histogram of %s pearson correlation on %s within individual vs between" % (n, k))
            plt.savefig(os.path.join(outpath, "hist_%s_on_%s_corr_5years_all.png" % (n, k)))
            plt.close("all")

            match = []
            all_mathced = True
            cols = list(r.columns)
            inds = list(r.index)
            print("For %s on %s" % (n, k))
            while len(r) > 0:
                match.append([r.max().max(), cols.index(r.max().idxmax()), inds.index(r.max(1).idxmax())])
                if match[-1][0] != r.loc[inds[match[-1][2]], cols[match[-1][1]]]:
                    print("WTF")
                    break
                if match[-1][2] != match[-1][1]:
                    all_mathced = False
                    print("non match %d %d" % (match[-1][2], match[-1][1]))
                r.drop(inds[match[-1][2]], inplace=True)
                r.drop(cols[match[-1][1]], axis=1, inplace=True)
            if all_mathced:
                print("Perfect gready match")

    for n in ['spearman', "log_fold", "fold"]:
        res[n] = pandas.DataFrame(res[n], index=['Mean', 'SD']).T
        res[n].to_csv(os.path.join(outpath, "stats_%s.csv" % n))
        labels = ["%s %s" % (x[0], x[1]) for x in res[n].index]
        plt.bar(range(len(res[n])), res[n].Mean.values, color="rb"*int(len(res[n])/2))
        plt.errorbar(range(len(res[n])), res[n].Mean.values, yerr=res[n].SD.values, linestyle="None", color="k")
        plt.xticks(range(len(res[n])), labels, rotation="vertical")
        if n != 'spearman':
            plt.title("Mean Value of pearson correlation of %s" % n)
        else:
            plt.title("Mean Value of spearman correlation of log_fold")
        plt.tight_layout()
        plt.savefig(os.path.join(outpath, "stats_%s.png" % n))
        plt.close("all")
