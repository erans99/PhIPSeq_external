import os
import numpy
import pandas
from scipy.stats import mannwhitneyu
from config import base_path, out_path
import matplotlib.pyplot as plt

MIN_OLIS = 200
THROW_BAD_OLIS = True


def score_groups(grps):
    sc = [{}, {}]
    for g1 in grps.keys():
        for g2 in grps.keys():
            sc[0][(g1 + "_less", g2)] = mannwhitneyu(grps[g1], grps[g2], alternative='less')[1]
            sc[1][(g1 + "_less", g2)] = \
                mannwhitneyu(grps[g1][grps[g1] > 0], grps[g2][grps[g2] > 0], alternative='less')[1]
    sc[0] = pandas.Series(sc[0]).unstack()
    sc[1] = pandas.Series(sc[1]).unstack()
    print("all\n", sc[0])
    print("non_0\n", sc[1])
    return sc


if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

    df_info = pandas.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0)
    df_info = df_info[(df_info.is_allergens | df_info.is_IEDB) & (df_info['num_copy'] == 1)]
    inds = df_info.index

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

    df_info['order'] = numpy.nan
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_IEDB & df_info.is_infect].sort_values('IEDB_organism_name').index
    df_info.loc[tmp_inds, 'order'] = range(len(tmp_inds))
    base = len(tmp_inds)
    pos = [['is_iedb_infect', base]]
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_IEDB & df_info.is_auto].sort_values('IEDB_organism_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_iedb_auto', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_IEDB].sort_values('IEDB_organism_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_iedb_other', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_plant].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_plant', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_animal].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_animal', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_insect].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_insect', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_fungi].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_fungi', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_human].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_human', base])
    tmp_inds = df_info[df_info['order'].isna() & df_info.is_allergens & df_info.is_bacteria].sort_values('allergens_common_name').index
    df_info.loc[tmp_inds, 'order'] = range(base, base+len(tmp_inds))
    base += len(tmp_inds)
    pos.append(['is_bacteria', base])

    df_info.sort_values('order', inplace=True)
    fold_df = pandas.concat([fold_df, pandas.DataFrame(columns=df_info.index.difference(fold_df.columns),
                                                       index=fold_df.index, data=1)], axis=1)[df_info.index]

    plt.figure()
    tmp = numpy.log(fold_df)
    plt.hist(tmp[df_info[df_info.is_allergens].index].values.flatten(), bins=numpy.arange(0, 6, 0.1),
             label="allergens (%d*%d)" % (df_info.is_allergens.sum(), len(tmp)), alpha=0.5)
    plt.hist(tmp[df_info[df_info.is_IEDB].index].values.flatten(), bins=numpy.arange(0, 6, 0.1),
             label="IEDB (%d*%d)" % (df_info.is_IEDB.sum(), len(tmp)), alpha=0.5)
    sc = score_groups({'allergens': tmp[df_info[df_info.is_allergens].index].values.flatten(),
                       'IEDB': tmp[df_info[df_info.is_IEDB].index].values.flatten()})
    plt.yscale('log')
    plt.legend()
    plt.title("Histogram of all log fold change of oligos in groups")
    plt.savefig(os.path.join(out_path, "hist_all_fold.png"))
    plt.close('all')

    df_info[['order', 'IEDB_organism_name', 'allergens_common_name', 'is_IEDB', 'is_auto', 'is_infect',
             'is_allergens', 'by_digestion', 'by_inhalation', 'by_skin', 'full name']].to_csv(
        os.path.join(out_path, "oligo_order.csv"))

    cs = {'is_plant': '#2e75b6fe',
          'is_animal': '#bdd7eefe',
          'is_insect': '#deebf7fe',
          'is_fungi': '#d0cecefe',
          'is_human': '#767171fe',
          'is_bacteria': '#181717fe',
          'is_iedb_auto': '#fbe5d6fe',
          'is_iedb_infect': '#f4b183fe',
          'is_iedb_other': '#c55a11fe'}

    plt.figure(figsize=[15, 5])
    old_p = 0
    ticks = [[], []]
    for name, p in pos:
        plt.plot(range(old_p, p), numpy.log(fold_df[fold_df.columns[old_p:p]].mean()), color=cs[name])
        ticks[0].append(name)
        ticks[1].append((old_p + p) / 2)
        old_p = p
    h = plt.ylim()[1]
    plt.title("Log of mean fold change")
    plt.xticks(ticks[1], ticks[0], rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(out_path, "fold_mean_color.png"))
    plt.close('all')

