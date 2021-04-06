import os
import pandas
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

lib = "agilent"
MIN_OLIS = 200
THROW_BAD_OLIS = True


def perform_dimensionality_reduction(out_path, existence_table, dimensionality_reduction_class, data_type,
                                     column_prefix, **kwargs):
    transformed_table = dimensionality_reduction_class.fit_transform(existence_table)
    transformed_table = pandas.DataFrame(index=existence_table.index,
                                         columns=list(map(lambda x: f'{column_prefix}{x}',
                                                          range(1, 1 + dimensionality_reduction_class.n_components))),
                                         data=transformed_table)
    transformed_table.to_csv(os.path.join(out_path, f'{data_type}.csv'))

    if 'pca' in data_type:
        pca_info = {}
        for i, c in enumerate(transformed_table.columns):
            pca_info[c] = dimensionality_reduction_class.explained_variance_ratio_[i]
        pca_info = pandas.Series(pca_info)
        pca_info.to_csv(os.path.join(out_path,  data_type + '_info.csv'))

    # Figure by status
    fig, axs = plt.subplots(ncols=3, nrows=2, figsize=(20,10))
    axs[-1, -1].axis('off')
    for i in range(5):
        x = f'{column_prefix}{i*2+1}'
        y = f'{column_prefix}{i*2+2}'
        ax = axs[int(i/3)][i % 3]
        if i != 4:
            sns.scatterplot(x=x, y=y, data=transformed_table, ax=ax, hue='Plate', legend=False)
        else:
            sns.scatterplot(x=x, y=y, data=transformed_table, ax=ax, hue='Plate')
            ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., ncol=2)
        if 'pca' in data_type:
            ax.set_xlabel('%s (explained variance %.2g%%)' % (x, 100 * pca_info.loc[x]))
            ax.set_ylabel('%s (explained variance %.2g%%)' % (y, 100 * pca_info.loc[y]))

    plt.suptitle("All PCs of %s" % data_type.replace("_", " "))
    plt.tight_layout()
    fig.savefig(os.path.join(out_path, f'{data_type}.png'))
    plt.close(fig)


def perform_pca(out_path, existence_table, data_type, n_components=10, **kwargs):
    n_components = min(n_components, existence_table.shape[1])
    pca = PCA(n_components=n_components)
    perform_dimensionality_reduction(out_path, existence_table, pca, data_type, 'PC', **kwargs)


if __name__ == '__main__':
    cache_path = os.path.join(base_path, "Cache")
    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent_final.pkl"))

    out_path = os.path.join(base_path, "PCAs")
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    inds = {}
    inds['bacterial'] = df_info[~df_info.is_IEDB_or_cntrl].index
    inds['microbiome'] = df_info[df_info.is_PNP | df_info.is_nonPNP_strains].index
    inds['PNP'] = df_info[df_info.is_PNP].index
    inds['VFDB'] = df_info[df_info.is_toxin].index

    metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s_above%d.pkl" % (lib, MIN_OLIS)))
    exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS)))
    fold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS)))

    fold_df.fillna(0, inplace=True)
    if THROW_BAD_OLIS:
        print("Throwing %d oligos" % len(fold_df.columns[(fold_df == -1).sum() > 0]))
        fold_df = fold_df[fold_df.columns[(fold_df == -1).sum() == 0]]
    else:
        fold_df = fold_df.replace(-1, 1)
    for k in inds.keys():
        perform_pca(out_path, fold_df[fold_df.columns.intersection(inds[k])], "%s_pca_lfold" % k)
    print()
