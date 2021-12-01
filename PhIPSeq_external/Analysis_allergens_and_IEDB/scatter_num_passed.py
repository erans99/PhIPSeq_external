import os
import numpy
import pandas
from scipy.stats import mannwhitneyu
import matplotlib.pyplot as plt
from config import base_path, out_path

MIN_OLIS = 200
THROW_BAD_OLIS = True

if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

    df_info = pandas.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)
    df_info = df_info[(df_info.is_allergens | df_info.is_IEDB) & (df_info['num_copy'] == 1)]
    inds = df_info.index

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
    exist_df = (fold_df > 1).astype(int)

    plt.scatter(exist_df[exist_df.columns.intersection(df_info[df_info.is_allergens].index)].sum(1),
                exist_df[exist_df.columns.intersection(df_info[df_info.is_IEDB].index)].sum(1),
                label="num oligos passed")
    num_olis = [len(df_info[df_info.is_allergens]), len(df_info[df_info.is_IEDB])]
    x = plt.xlim()[1]
    y = x * num_olis[1] / num_olis[0]
    plt.plot([0, x], [0, y], color='r', linestyle='--', label="num expected by group size")
    plt.xlabel("num allergen oligos passed")
    plt.ylabel("num IEDB oligos passed")
    plt.legend()
    sc = mannwhitneyu(exist_df[exist_df.columns.intersection(df_info[df_info.is_allergens].index)].sum(1) / num_olis[0],
                      exist_df[exist_df.columns.intersection(df_info[df_info.is_IEDB].index)].sum(1) / num_olis[1])
    plt.title("Number of oligos passed per person, for the two major groups\nMann-Whitney stat %g pval %g" %
              (sc[0], sc[1]))
    plt.savefig(os.path.join(out_path, "scatter_passed_IEDB_vs_allergens.png"))

    print(sc)
    print()
