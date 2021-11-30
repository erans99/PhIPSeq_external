import os

import matplotlib.pyplot as plt
import numpy
import pandas
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests

from config import base_path, out_path

MIN_OLIS = 200
THROW_BAD_OLIS = True

MIN_EXIST_OLIS = 0.02

if __name__ == '__main__':
    os.makedirs(out_path, exist_ok=True)

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

    base = meta_df.index[1::2]
    repeat = meta_df.index[::2]

    base_exist = exist_df.loc[base].copy().astype(int)
    base_exist.index = meta_df.loc[base].ID
    repeat_exist = exist_df.loc[repeat].copy().astype(int)
    repeat_exist.index = meta_df.loc[repeat].ID

    cols = exist_df.columns[exist_df.sum() > (MIN_EXIST_OLIS * len(exist_df))]
    print(len(cols))
    lcols = len(cols)

    plt.scatter(base_exist[cols].sum(), repeat_exist[cols].sum())
    m = plt.xlim()[1]
    plt.plot([0, m], [0, m], color='r')
    plt.title("Number of people exist in before and after")
    plt.xlabel("Before")
    plt.ylabel("After")
    plt.savefig(os.path.join(out_path, "scatter_num_exist.png"))
    plt.close("all")

    base_exist = base_exist[cols]
    repeat_exist = repeat_exist[cols]

    lind = len(base_exist)
    chng2 = (repeat_exist * 2 + base_exist)
    prob_chng = [(chng2 == 0).sum().sum() / (base_exist == 0).sum().sum(),
                 (chng2 == 1).sum().sum() / (base_exist == 1).sum().sum()]
    chi = {}
    for x in chng2.columns:
        f_obs = []
        st = [(base_exist[x] == 0).sum()]
        st.append(lind - st[0])
        f_exp = [st[0] * prob_chng[0], st[1] * prob_chng[1], st[0] * (1 - prob_chng[0]), st[1] * (1 - prob_chng[1])]
        for i in range(4):
            f_obs.append((chng2[x] == i).sum())
        chi[x] = list(chisquare(f_obs, f_exp, 1)) + [st, f_obs[0] + f_obs[3], f_obs[1] + f_obs[2]]
    chi = pandas.DataFrame(chi, index=['chi_stat', 'chi_pval', 'base_react', 'num_stable', 'num_changed']).T
    chi = chi.dropna()

    FDR = multipletests(chi.chi_pval.values.tolist(), method='fdr_by')
    chi['passed_FDR'] = FDR[0]
    chi['qval'] = FDR[1]
    chi['passed_Bonf'] = chi.chi_pval < (0.05 / len(chi))

    print("%d passed FDR %d Bonf" % (len(chi[chi.passed_FDR]), len(chi[chi.passed_Bonf])))
    chi.sort_values('chi_pval', inplace=True)

    df_passed = chi[chi.passed_FDR].copy()
    df_passed['non_stable'] = df_passed.num_stable < (lind * (2 * prob_chng[0] - 1))
    df_passed['oligo_name'] = df_info.loc[df_passed.index]['full name'].values
    add_cols = ['is_IEDB', 'is_allergens', 'is_auto', 'is_infect', 'by_skin', 'by_digestion', 'by_inhalation',
                'is_IEDB_other']
    cnt = {}
    for c in add_cols[:-1]:
        df_passed[c] = df_info.loc[df_passed.index][c].values
        cnt[c] = [df_info[c].sum(), df_info.loc[cols][c].sum()]
    df_passed['is_auto_not_infect'] = df_passed['is_IEDB'] & (df_passed['is_auto']) & (~df_passed['is_infect'])
    cnt['is_auto_not_infect'] = [((df_info['is_auto']) & (~df_info['is_infect'])).sum(),
                                 ((df_info.loc[cols]['is_auto']) & (~df_info.loc[cols]['is_infect'])).sum()]
    df_passed['is_IEDB_other'] = df_passed['is_IEDB'] & (~df_passed['is_auto']) & (~df_passed['is_infect'])
    cnt['is_IEDB_other'] = [(df_info['is_IEDB'] & (~df_info['is_auto']) & (~df_info['is_infect'])).sum(),
                            (df_info.loc[cols]['is_IEDB'] & (~df_info.loc[cols]['is_auto']) &
                             (~df_info.loc[cols]['is_infect'])).sum()]
    cnt = pandas.DataFrame(cnt, index=['all', 'tested'])

    df_passed.to_csv(os.path.join(out_path, "FDR_changed.csv"))

    df_passed = df_passed[df_passed.non_stable]
    print("Non stable: %d passed FDR %d Bonf" % (len(df_passed), len(df_passed[df_passed.passed_Bonf])))
    check_cols = ['is_infect', 'is_auto_not_infect', 'is_IEDB_other', 'by_skin', 'by_digestion', 'by_inhalation']

    fig, ax = plt.subplots(figsize=[10, 5])
    pos = numpy.arange(len(check_cols))
    w = 0.3
    rects1 = ax.bar(pos, df_passed[check_cols].sum(), w, label='num passed FDR', color='#ffd966fe')
    rects2 = ax.bar(pos + w, cnt.loc['tested'][check_cols] / sum(cnt.loc['tested'][check_cols]) *
                    df_passed[check_cols].sum().sum(), w,
                    label='expected by num_checked', color='#a9d18efe')
    plt.legend()
    plt.xticks(pos, check_cols)
    plt.title("Number of oligos which passed FDR for non-stability in different groups")
    plt.savefig(os.path.join(out_path, "FDR_changed_by_type.png"))
    plt.close("all")
