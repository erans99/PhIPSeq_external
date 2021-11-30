import os

import matplotlib.pyplot as plt
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats import mannwhitneyu
import numpy
import pandas
from config import base_path, out_path

MIN_OLIS = 200
THROW_BAD_OLIS = True

MIN_EXIST_OLIS = 0.02  # reactive oligos appear in above this percent of samples
NUM_WITH_MIN_EXIST_OLIS = 100  # group needs to have at least 100 reactive oligos


def run_on_inds(k, inds, base, repeat, outpath, fold_df):
    tmp_fold_df = fold_df[fold_df.columns[(fold_df > 1).sum() > (MIN_EXIST_OLIS * len(fold_df))]]
    inds = tmp_fold_df.columns.intersection(inds)
    print("%s: %d inds" % (k, len(inds)))

    res = {}
    for i, b in enumerate(base):
        if (i % 10) == 0:
            print("At %d or %d" % (i, len(base)))
        tmp = {}
        for j, r in enumerate(repeat):
            comp = tmp_fold_df.loc[[b, r], inds].dropna(axis=1, how='all').T
            comp.fillna(1, inplace=True)
            tmp[r] = comp.corr('spearman').loc[b, r]
        res[b] = tmp

    res = pandas.DataFrame(res).T
    res = res.loc[base, repeat]
    res.to_csv(os.path.join(outpath, "spearman_on_%s_corr_5years_all.csv" % k))


def plot_hist(r, outpath, k, meta_df, base, repeat):
    plt.figure()
    plt.imshow(r.values, cmap='plasma', vmin=0, vmax=1)
    plt.xlim(0, len(r))
    plt.ylim(0, len(r))

    plt.xticks(range(0, len(r), 10),
               [str(meta_df.loc[x].ID) + " " + str(meta_df.loc[x].Date.split("-")[0]) for x in base[::10]],
               rotation='vertical')
    plt.yticks(range(0, len(r), 10),
               [str(meta_df.loc[x].ID) + " " + str(meta_df.loc[x].Date.split("-")[0]) for x in repeat[::10]],
               rotation='horizontal')
    cbar = plt.colorbar()
    cbar.ax.set_ylabel("spearman correlation", rotation=-90, va="bottom")
    plt.title("spearman correlation on %s of old and new samples\nDiagonal is same person after 5 years" % k)
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "spearman_on_%s_corr_5years_all.png" % k))
    plt.close("all")


def print_match_info(r, k):
    match = []
    all_mathced = True
    cols = list(r.columns)
    inds = list(r.index)
    print("For spearman on %s" % k)
    while len(r) > 0:
        if (~r.isna()).sum().sum() == 0:
            print("All the rest of the matrix (%d) are nans... Skipping" % len(r))
            break
        try:
            m = r.max().max()
            c = cols.index(r.max().idxmax())
            i = inds.index(r[cols[c]].idxmax())
            match.append([m, c, i])
        except:
            print()
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


def plot_box(res, outpath, name):
    cat_ord = res.index.get_level_values(0).unique().to_list()
    tmp = pandas.concat({x: pandas.Series(res.loc[x].vals + [0] * res.loc[x].num_miss)
                         for x in res.index}).reset_index()
    tmp.columns = ['category', 'grp', 'junk', 'corr']
    plt.figure(constrained_layout=True, figsize=(20, 10))
    fig = sns.boxplot(data=tmp, x='category', y='corr', hue='grp', order=cat_ord,
                      palette=['#8ebad9fe', '#ea9293fe'])
    box_pairs = [((k, "between"), (k, "within")) for k in cat_ord]
    add_stat_annotation(fig, data=tmp, x='category', y='corr', hue='grp', order=cat_ord,
                        box_pairs=box_pairs, test='Mann-Whitney', text_format='star', loc='inside', verbose=2)
    plt.xticks(range(int(len(res) / 2)), plt.xticks()[1], rotation='vertical')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xlabel(None)
    plt.tight_layout()
    plt.savefig(os.path.join(outpath, "%s_box_corrs.png" % name))
    plt.close('all')


if __name__ == '__main__':
    df_info = pandas.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)
    df_info = df_info[(df_info.is_allergens | df_info.is_IEDB) & (df_info['num_copy'] == 1)]
    inds = df_info.index
    l_base = len(inds)

    meta_df = pandas.read_csv(os.path.join(base_path, "cohort.csv"), index_col=0, low_memory=False)
    vs = meta_df.ID.value_counts()
    vs = vs[vs == 2].index
    meta_df = meta_df[meta_df.ID.isin(vs)].sort_values(['ID', 'timepoint'])

    fold_df = pandas.read_csv(os.path.join(base_path, "fold_data.csv"), index_col=[0, 1], low_memory=False).loc[
        meta_df.index].unstack()
    fold_df.columns = fold_df.columns.get_level_values(1)
    fold_df = fold_df[fold_df.columns.intersection(inds)]

    if THROW_BAD_OLIS:
        drop = fold_df.columns[(fold_df == -1).sum() > 0]
        fold_df = fold_df[fold_df.columns.difference(drop)].fillna(1)
        inds = df_info.index.difference(drop)
        df_info = df_info.loc[inds]
    exist_df = (fold_df > 1).astype(int)

    for state in ['All', 'IEDB', 'Allergens']:
        print(state)
        inds = {}
        if state == 'All':
            name = "all"
            outpath = out_path
            inds['All'] = fold_df.columns.intersection(df_info.index)
            for n in ['is_IEDB', 'is_allergens']:
                inds[n] = fold_df.columns.intersection(df_info[df_info[n]].index)
            for k in inds.keys():
                print(k, len(inds[k]), ((fold_df[inds[k]] > 1).sum() > (MIN_EXIST_OLIS * len(fold_df))).sum())
        elif state == 'IEDB':
            name = "IEDB_type"
            outpath = os.path.join(out_path, "IEDB")
            tmp = set(fold_df.columns.intersection(df_info[df_info.is_IEDB].index))
            for n in ['is_infect', 'is_auto']:
                inds[n] = fold_df.columns.intersection(df_info[df_info.is_IEDB & df_info[n]].index)
                tmp = tmp.difference(inds[n])
            inds['other_IEDB'] = tmp
        else:
            name = "allenrgen_by"
            outpath = os.path.join(out_path, "Allergens")
            for n in ['by_digestion', 'by_inhalation', 'by_skin']:
                inds[n] = fold_df.columns.intersection(df_info[df_info.is_allergens & df_info[n]].index)
        os.makedirs(outpath, exist_ok=True)

        base = meta_df.index[1::2]
        repeat = meta_df.index[::2]

        for k in inds.keys():
            if (((fold_df[inds[k]] > 1).sum() > (MIN_EXIST_OLIS * len(fold_df))).sum()) < NUM_WITH_MIN_EXIST_OLIS:
                print("Can't work with %s, not enough info" % k)
                continue
            if not os.path.exists(os.path.join(outpath, "%s_on_%s_corr_5years_all.csv" % ("spearman", k))):
                run_on_inds(k, inds[k], base, repeat, outpath, fold_df)

        res = {}
        med = {}
        for k in inds.keys():
            print("Working on spearman on %s" % k)
            try:
                r = pandas.read_csv(os.path.join(outpath, "spearman_on_%s_corr_5years_all.csv" % k), index_col=0,
                                    low_memory=False)
            except:
                print("No info for spearman %s. Skipping" % k)
                continue

            if not os.path.exists(os.path.join(outpath, "spearman_on_%s_corr_5years_all.png" % k)):
                plot_hist(r, outpath, k, meta_df, base, repeat)

            vals = [[], []]
            miss = [0, 0]
            med_diff = []
            s = len(r.columns)
            for i in range(len(r)):
                if numpy.isnan(r.iloc[i, i]):
                    miss[0] += 1
                else:
                    vals[0].append(r.iloc[i, i])
                v = r.iloc[i, :i].dropna().values.tolist() + r.iloc[i, i + 1:].dropna().values.tolist()
                vals[1] += v
                miss[1] += s - 1 - len(v)
                med_diff.append(vals[0][-1] - numpy.median(v + [0] * (s - 1 - len(v))))
                if r.iloc[i, i] != r.iloc[i].max():
                    print("Matched not best for:", i, r.iloc[i].name, r.T.iloc[i].name)
            res[(k, 'between')] = [numpy.array(vals[1] + [0] * miss[1]).mean(),
                                   numpy.array(vals[1] + [0] * miss[1]).std(),
                                   miss[1], vals[1]]
            res[(k, 'within')] = [numpy.array(vals[0] + [0] * miss[0]).mean(),
                                  numpy.array(vals[0] + [0] * miss[0]).std(),
                                  miss[0], vals[0]]
            med[k] = [numpy.array(med_diff).mean(), numpy.array(vals[0]).std(), med_diff]

            print_match_info(r, k)

        res = pandas.DataFrame(res, index=['Mean', 'SD', 'num_miss', 'vals']).T
        res[['Mean', 'SD', 'num_miss']].to_csv(os.path.join(outpath, "%s_stats_%s.csv" % (name, n)))
        if ('is_IEDB' in inds.keys()) and ('is_allergens' in inds.keys()):
            print("For %s IEDB vs allergens Mann-Whitney on diff within and median between:" % n)
            print(mannwhitneyu(med['is_allergens'][2], med['is_IEDB'][2]))
        plot_box(res, outpath, name)
