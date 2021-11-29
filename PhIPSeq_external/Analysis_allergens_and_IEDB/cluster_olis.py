import matplotlib.patches as patches
import os
import numpy
import pandas
from matplotlib.gridspec import GridSpec
from scipy.cluster.hierarchy import fcluster, linkage, dendrogram
from scipy.spatial.distance import squareform
import seaborn as sns
import matplotlib.pyplot as plt
from config import base_path, out_path

MIN_OLIS = 200
THROW_BAD_OLIS = True
MIN_APPEAR = 0.02
MIN_CLUST = 10
CLUST_TH = 0.7


def get_clusters(link, dn, inds, th=CLUST_TH):
    clst = fcluster(link, criterion='distance', t=th)
    return pandas.Series(index=inds, data=clst).iloc[dn['leaves']]


def get_groups(clst, ax=None, clust_above=MIN_CLUST, fout=None, color_dict=None):
    line_params = {'color': 'white', 'linewidth': 1}
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
            if ax is not None:
                if color_dict is not None:
                    line_params['color'] = color_dict[str(len(groups))][1]
                    ax.text(groups[-1][1] + 4, groups[-1][0] - 4, color_dict[str(len(groups))][0],
                            color=color_dict[str(len(groups))][1], fontsize=14)
                else:
                    ax.text(groups[-1][1] + 4, groups[-1][0] - 4, str(len(groups)), color='white', fontsize=14)
                ax.vlines(groups[-1][0], groups[-1][0], groups[-1][1], **line_params)
                ax.vlines(groups[-1][1], groups[-1][0], groups[-1][1], **line_params)
                ax.hlines(groups[-1][0], groups[-1][0], groups[-1][1], **line_params)
                ax.hlines(groups[-1][1], groups[-1][0], groups[-1][1], **line_params)

            groups.append([i])
        else:
            groups[-1][0] = i
        v = clst[i]
    groups = groups[:-1]
    if fout is not None:
        out_csv = []
        for i, g in enumerate(groups):
            out_csv.append(pandas.DataFrame(clst.iloc[g[0]:g[1]].index, columns=['oligos']))
            out_csv[-1]['group'] = 'Oligos_%d' % i
        pandas.concat(out_csv, ignore_index=True).to_csv(fout)
    return groups


def add_groups(groups, ax, is_v=True):
    y_values = ax.get_ylim()
    x_values = ax.get_xlim()
    start = 0
    for g in groups:
        if is_v:
            rect = patches.Rectangle((start, 0), g[0] - start, y_values[0], linewidth=1, edgecolor='white',
                                     facecolor='white', alpha=0.3)
        else:
            rect = patches.Rectangle((0, start), x_values[1], g[0] - start, linewidth=1, edgecolor='white',
                                     facecolor='white', alpha=0.3)
        ax.add_patch(rect)
        start = g[1]

    if is_v:
        rect = patches.Rectangle((start, 0), x_values[1] - start, y_values[0], linewidth=1, edgecolor='white',
                                 facecolor='white', alpha=0.3)
    else:
        rect = patches.Rectangle((0, start), x_values[1], y_values[0] - start, linewidth=1, edgecolor='white',
                                 facecolor='white', alpha=0.3)

    ax.add_patch(rect)


if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

    df_info = pandas.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0)
    df_info = df_info[(df_info.is_allergens | df_info.is_IEDB) & (df_info['num_copy'] == 1)]
    inds = df_info.index
    l_base = len(inds)

    meta_df = pandas.read_csv(os.path.join(base_path, "cohort.csv"), index_col=0)
    meta_df = meta_df[(meta_df.timepoint == 1) & (meta_df.num_passed >= MIN_OLIS)]

    fold_df = pandas.read_csv(os.path.join(base_path, "fold_data.csv"), index_col=[0, 1]).loc[meta_df.index].unstack()
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
    inds_algn = df_info[df_info.is_allergens].index
    inds_IEDB = df_info[df_info.is_IEDB].index

    th = CLUST_TH

    print("Clustering allergens", fold_df[inds_algn].shape)
    corr = fold_df[inds_algn].corr('spearman')
    link = linkage(squareform(1 - corr), method='average')
    dn = dendrogram(link, no_plot=True)
    clst = get_clusters(link, dn, corr.columns, th)

    fig = plt.figure(figsize=[8.4, 10])
    gs = GridSpec(2, 2, height_ratios=[4, 1], width_ratios=[20, 1], hspace=0.1)

    dendogram_ax = fig.add_subplot(gs[0])
    dendogram_ax.text(-0.1, 1.0, "A", transform=dendogram_ax.transAxes, size=20, weight='bold')
    cbar_ax = fig.add_subplot(gs[1])
    sns.heatmap(corr.iloc[dn['leaves'], dn['leaves']], ax=dendogram_ax, yticklabels=False, xticklabels=False,
                cmap=sns.color_palette('viridis', as_cmap=True), cbar_ax=cbar_ax)
    print(clst.max())
    pandas.Series(corr.index[dn['leaves']]).to_csv(os.path.join(out_path, "all_allergen_oligo_order.csv"))
    groups = get_groups(clst, dendogram_ax, fout=os.path.join(out_path, "all_allergen_oligo_groups_%g.csv" %
                                                              MIN_APPEAR))
    print("%d oligo groups" % len(groups))

    bar_axis = fig.add_subplot(gs[2], sharex=dendogram_ax)
    use_columns = ['is_animal',
                   'is_bacteria',
                   'is_fungi',
                   'is_human',
                   'is_insect',
                   'is_plant',
                   'is_food',
                   ]
    sns.heatmap(data=df_info.loc[inds_algn][use_columns].astype(float).iloc[dn['leaves']].T * 1, xticklabels=False,
                yticklabels=list(map(lambda col: col.split('_')[1].capitalize(), use_columns)), ax=bar_axis,
                cmap=sns.color_palette('viridis'), cbar=False)
    bar_axis.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)

    add_groups(groups, bar_axis)
    plt.savefig(os.path.join(out_path, "cluster_olis_allergens_above_%g.png" % MIN_APPEAR))

    print("Clustering IEDB", fold_df[inds_IEDB].shape)
    corr = fold_df[inds_IEDB].corr('spearman')
    link = linkage(squareform(1 - corr), method='average')
    dn = dendrogram(link, no_plot=True)
    clst = get_clusters(link, dn, corr.columns, th)

    fig = plt.figure(figsize=[11.5, 10])
    gs = GridSpec(1, 3, width_ratios=[20, 2, 1], hspace=0.1)

    dendogram_ax = fig.add_subplot(gs[0])
    dendogram_ax.text(-0.1, 1.0, "A", transform=dendogram_ax.transAxes, size=20, weight='bold')
    cbar_ax = fig.add_subplot(gs[2])
    sns.heatmap(corr.iloc[dn['leaves'], dn['leaves']], ax=dendogram_ax, yticklabels=False, xticklabels=False,
                cmap=sns.color_palette('viridis', as_cmap=True), cbar_ax=cbar_ax)
    print(clst.max())
    pandas.Series(corr.index[dn['leaves']]).to_csv(os.path.join(out_path, "all_IEDB_oligo_order.csv"))
    groups = get_groups(clst, dendogram_ax, fout=os.path.join(out_path, "all_IEDB_oligo_groups_%g.csv" % MIN_APPEAR))
    print("%d oligo groups" % len(groups))

    bar_axis = fig.add_subplot(gs[1], sharey=dendogram_ax)
    use_columns = ['is_auto', 'is_infect']
    sns.heatmap(data=df_info.loc[inds_IEDB][use_columns].astype(float).iloc[dn['leaves']] * 1, yticklabels=False,
                xticklabels=list(map(lambda col: col.split('_')[1].capitalize(), use_columns)), ax=bar_axis,
                cmap=sns.color_palette('viridis'), cbar=False)
    bar_axis.tick_params(right=True, labelright=True, left=False, labelleft=False, rotation=0)

    add_groups(groups, bar_axis, is_v=False)

    plt.savefig(os.path.join(out_path, "cluster_olis_IEDB_above_%g.png" % MIN_APPEAR))

    plt.close('all')
