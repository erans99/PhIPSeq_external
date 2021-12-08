from scipy.stats import mannwhitneyu
import matplotlib.patches as patches
import os
import numpy
import pandas
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
from scipy.stats import spearmanr
from statsmodels.stats.multitest import multipletests
import matplotlib.pyplot as plt

from config import base_path, out_path

MIN_OLIS = 200
THROW_BAD_OLIS = True
MIN_APPEAR = 0.02
CLUST_TH = 0.7
MIN_CLUST = 10


def get_clusters(link, dn, inds, th=0.7):
    clst = fcluster(link, criterion='distance', t=th)
    return pandas.Series(index=inds, data=clst).iloc[dn['leaves']]


def draw_significant_groups(groups, dn_ax, color='white'):
    # Draw boxes around clusters
    for group in groups:
        rect = patches.Rectangle((group[0][0], group[1][0]), group[0][1] - group[0][0], group[1][1] - group[1][0],
                                 linewidth=1, edgecolor=color, facecolor='none')
        dn_ax.add_patch(rect)


def draw_legume_group(group, ax):
    y_values = ax.get_ylim()
    x_values = ax.get_xlim()
    rect = patches.Rectangle((0, 0), x_values[1], group[0], linewidth=1, edgecolor='white',
                             facecolor='white', alpha=0.6)
    ax.add_patch(rect)
    rect = patches.Rectangle((0, group[1]), x_values[1], y_values[0] - group[1], linewidth=1, edgecolor='white',
                             facecolor='white', alpha=0.6)
    ax.add_patch(rect)


def get_groups(clst, clust_above=MIN_CLUST):
    groups = []
    v = -1
    for i in range(len(clst)):
        if clst[i] == v:
            continue
        if v == -1:
            groups.append([i])
            v = clst[i]
            continue
        if (i - groups[-1][0]) >= clust_above:
            groups[-1].append(i)
            groups.append([i])
        else:
            groups[-1][0] = i
        v = clst[i]
    groups = groups[:-1]
    return groups


if __name__ == "__main__":
    os.makedirs(out_path, exist_ok=True)

    df_info = pandas.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)
    df_info = df_info[df_info.is_allergens & (df_info['num_copy'] == 1)]
    inds = df_info.index
    l_base = len(inds)

    meta_df = pandas.read_csv(os.path.join(base_path, "cohort.csv"), index_col=0, low_memory=False)
    meta_df = meta_df[(meta_df.timepoint == 1) & (meta_df.num_passed >= MIN_OLIS)]

    fold_df = pandas.read_csv(os.path.join(base_path, "fold_data.csv"), index_col=[0, 1],
                              low_memory=False).loc[meta_df.index].unstack()
    fold_df.columns = fold_df.columns.get_level_values(1)
    fold_df = fold_df[fold_df.columns.intersection(inds)]

    if THROW_BAD_OLIS:
        drop = fold_df.columns[(fold_df == -1).sum() > 0]
        fold_df = fold_df[fold_df.columns.difference(drop)].fillna(1)
        inds = df_info.index.difference(drop)
        df_info = df_info.loc[inds]

    fold_df = fold_df[fold_df.columns[(fold_df > 1).sum() > (MIN_APPEAR * len(fold_df))]]
    fold_df = numpy.log(fold_df.fillna(1))
    df_info = df_info.loc[fold_df.columns]

    th = CLUST_TH

    # Oligos level correlations
    corr = fold_df.corr('spearman')
    link = linkage(squareform(1 - corr), method='average')
    dn = dendrogram(link, no_plot=True)
    clst = get_clusters(link, dn, corr.columns, th)
    groups = get_groups(clst)

    # Samples level correlations
    corr1 = fold_df.T.corr('spearman')
    link1 = linkage(squareform(1 - corr1), method='average')
    dn1 = dendrogram(link1, no_plot=True)
    clst1 = get_clusters(link1, dn1, corr1.columns, th)
    groups1 = get_groups(clst1)

    # Define figure
    fig = plt.figure(figsize=[9.2, 12])
    gs = GridSpec(1, 3, width_ratios=[0.2, 3, 1])

    # Plot heatmap
    bar_ax = fig.add_subplot(gs[0])
    dendogram_ax = fig.add_subplot(gs[1])
    sns.heatmap(fold_df.iloc[dn1['leaves'], dn['leaves']], cmap=sns.color_palette('flare', as_cmap=True),
                ax=dendogram_ax, yticklabels=False, xticklabels=False, cbar_ax=bar_ax)

    dendogram_ax.set_xlabel("oligos")
    dendogram_ax.set_ylabel("samples")

    # Plot sample level bars
    mt = 'normalized mt_1342'
    bar_axis1 = fig.add_subplot(gs[2], sharey=dendogram_ax)
    meta_df['yob'] = (meta_df['yob'] - 1944) / 60
    use_columns = ['gender', 'yob']
    sample_extra_info = pandas.merge(meta_df[use_columns], meta_df[mt], left_index=True,
                                     right_index=True, how='left')
    sample_extra_info[mt] = ((sample_extra_info[mt] - sample_extra_info[mt].min()) /
                             (sample_extra_info[mt].max() - sample_extra_info[mt].min())).astype(float)
    sample_extra_info.rename(columns={mt: 'norm mt_1342'}, inplace=True)
    mt = 'norm mt_1342'
    sample_extra_info = sample_extra_info.iloc[dn1['leaves']]
    sns.heatmap(data=sample_extra_info, xticklabels=sample_extra_info.columns, yticklabels=False,
                ax=bar_axis1, cmap=sns.color_palette("viridis", as_cmap=True))

    # Compute significant shared groups
    fold_df = fold_df.iloc[dn1['leaves'], dn['leaves']].copy()
    significant_groups = []
    for oligo_subgroup in groups:
        sample_group_means = sorted(enumerate(
            [fold_df.iloc[range(*sample_group), range(*oligo_subgroup)].mean().mean() for sample_group in groups1]),
            key=lambda x: -x[1])
        if sample_group_means[0][1] > 2 * sample_group_means[1][1]:
            significant_groups.append([oligo_subgroup, groups1[sample_group_means[0][0]]])
    draw_significant_groups(significant_groups, dendogram_ax)

    mt_scores = pandas.Series([mannwhitneyu(sample_extra_info.iloc[range(*sample_group)][mt].dropna(),
                                            sample_extra_info.iloc[list(range(0, sample_group[0])) +
                                                                   list(range(sample_group[1], len(sample_extra_info)))]
                                            [mt].dropna())[1]
                               for oligos_group, sample_group in significant_groups])
    mt_group = significant_groups[mt_scores.idxmin()]
    mt_pval = mt_scores.min()
    draw_significant_groups([mt_group], dendogram_ax, color='blue')
    draw_legume_group(mt_group[1], bar_axis1)
    plt.suptitle('For group marked in blue the %s level\nof samples in group vs those not in group\n' % mt +
                 'got MW p-value of %g' % mt_pval)

    plt.savefig(os.path.join(out_path, "legumes.png"))

    res = {}
    inds = sample_extra_info[mt].dropna().index
    for i in range(*mt_group[0]):
        col = fold_df.columns[i]
        res[col] = spearmanr(sample_extra_info.loc[inds][mt], fold_df.loc[inds, col].values)
    res = pandas.DataFrame(res, index=['stat', 'pval']).T.sort_values('pval')
    res["Bonf"] = res['pval'] * len(res)
    FDR = multipletests(res.pval.values.tolist(), method='fdr_by')
    res["FDR_BY"] = FDR[0]
    res['FDR_BY_qval'] = FDR[1]
    FDR = multipletests(res.pval.values.tolist(), method='fdr_bh')
    res["FDR_BH"] = FDR[0]
    res['FDR_BH_qval'] = FDR[1]
    res['allergens_common_name'] = df_info.loc[res.index].allergens_common_name

    print("Of %d oligos in the blue group %d pass FDR (BY) vs %s" % (len(res), len(res[res.FDR_BY]), mt))
    res.to_csv(os.path.join(out_path, "mt_1342.csv"))

