import glob
import os
import sys
import numpy
import pandas
import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests

import scipy.stats

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

lib = "agilent"
MIN_OLIS = 200
THROW_BAD_OLIS = True
MIN_APPEAR_IN = 0.02


def corr_job(MB_df_part, lfold_df, out_file):
    res = []
    for SGB in MB_df_part.columns:
        for oli in lfold_df.columns:
            res.append([SGB.split("|")[-1], oli] + list(scipy.stats.spearmanr(MB_df_part[SGB], lfold_df[oli])))
    res = pandas.DataFrame(res, columns=['SGB', 'oligo', 'spearman_corr', 'spearman_pval'])
    res.to_csv(out_file)


if __name__ == '__main__':
    cache_path = os.path.join(base_path, "Cache")
    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent.pkl"))
    df_info['is_IEDB_or_cntrl'] = df_info.is_IEDB | df_info.is_pos_cntrl | df_info.is_neg_cntrl | df_info.is_rand_cntrl

    out_path = os.path.join(base_path, "MB_to_olis")
    tmp_out_path = os.path.join(out_path, "tmp")
    if not os.path.exists(tmp_out_path):
        os.makedirs(tmp_out_path)

    metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s_above%d.pkl" % (lib, MIN_OLIS))).T
    lfold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS))).T

    olis = lfold_df.columns[(lfold_df > 0).sum() > (MIN_APPEAR_IN * len(lfold_df))]
    lfold_df = lfold_df[olis]

    MB_df = pandas.read_pickle(os.path.join(cache_path, "MB_agilent_above200.pkl")).T
    SGBs = MB_df.columns[(MB_df > 0.0001).sum() > (MIN_APPEAR_IN * len(MB_df))]
    SGB_appear = (MB_df > 0.0001).sum()
    SGB_appear.index = [x.split("|")[-1] for x in SGB_appear.index]
    MB_df = numpy.log(MB_df[SGBs])

    lfold_df = lfold_df.loc[MB_df.index]

    if not os.path.exists(os.path.join(out_path, "all_corrs_MBPh.csv")):
        for i, st in enumerate(range(0, len(MB_df.columns), 10)):
            cols = MB_df.columns[st: st+10]
            corr_job(MB_df[cols], lfold_df, os.path.join(tmp_out_path, "part_%d.csv" % i))

        fs = glob.glob(os.path.join(tmp_out_path, "part_*.csv"))
        dfs = []
        for f in fs:
            dfs.append(pandas.read_csv(f, index_col=0))
        df = pandas.concat(dfs, ignore_index=True)
        df.to_csv(os.path.join(out_path, "all_corrs_MBPh.csv"))
        drop = df_info[df_info.is_IEDB_or_cntrl].index
        df_part = df[~numpy.isin(df.oligo, drop)]
        df_part.to_csv(os.path.join(out_path, "all_nonCNTRL_corrs_MBPh.csv"))
        df_full = df
    else:
        df_full = pandas.read_csv(os.path.join(out_path, "all_corrs_MBPh.csv"), index_col=0)
        df_part = pandas.read_csv(os.path.join(out_path, "all_nonCNTRL_corrs_MBPh.csv"), index_col=0)

    for df, name in [[df_full, ""], [df_part, "_nonCNTRL"]]:
        FDR = multipletests(df.spearman_pval.values.tolist(), method='fdr_bh')
        df['passed_FDR'] = FDR[0]
        df['qval'] = FDR[1]

        df_passed = df[df.passed_FDR].copy()
        df_passed.sort_values('qval', inplace=True)
        df_passed['num_oligo_appear_in'] = (lfold_df[df_passed.oligo] > 0).sum().values
        df_passed['num_SGB_appear_in'] = SGB_appear.loc[df_passed.SGB].values
        df_passed['oligo_name'] = df_info.loc[df_passed.oligo.values]['full name'].values
        df_passed['oligo_bac_src'] = df_info.loc[df_passed.oligo.values]['bac_src'].values
        df_passed['oligo_uniref_func'] = df_info.loc[df_passed.oligo.values]['uniref_func'].values

        d_SGBs = {x.split("|")[-1]: x.split("|")[:7] for x in SGBs}
        d_SGBs = pandas.DataFrame(d_SGBs, index=['k', 'p', 'c', 'o', 'f', 'g', 's']).T

        for c in ['Species', 'Genus', 'Family']:
            df_passed[c] = d_SGBs.loc[df_passed.SGB.values][c[0].lower()].values
        df_passed.to_csv(os.path.join(out_path, "passed%s_corrs_MBPh.csv" % name))

        df_passed = df_passed.sort_values('qval', ascending=False)
        plt.figure()
        plt.scatter(df_passed.num_oligo_appear_in, df_passed.num_SGB_appear_in, c=numpy.log10(df_passed.qval),
                    alpha=0.7)
        plt.xlabel("Number of individuals oligo appears in")
        plt.ylabel("Number of individuals SGB appears in")
        cbar = plt.colorbar()
        cbar.set_label("Log10 of FDR corrected p-value of correlation")
        cbar.ax.set_yticklabels([10 ** i for i in range(-7, -1)])
        plt.title("Strength of correlation between pairs\nof oligos and SGBs passing FDR correction")
        plt.savefig(os.path.join(out_path, "scatter_passed_FDR%s.png" % name))
        plt.close('all')

        vals = df_passed.SGB.value_counts()
        plt.figure(figsize=(7, 7))
        plt.hist(list(vals) + ([0] * (len(SGBs) - len(vals))), bins=range(vals.max()+1), color='#fecd30')
        plt.title("Histogram of SGBs reactivity")
        plt.ylabel("Number of SGBs")
        plt.xlabel("Number of oligos passed vs. each SGB")
        d2_SGBs = pandas.Series({x.split("|")[-1]: x for x in SGBs})
        for v in vals.index[:15]:
            print(v, vals.loc[v], d2_SGBs[v])
        plt.savefig(os.path.join(out_path, "hist_passed_SGBs%s.png" % name))
        plt.yscale('log')
        plt.savefig(os.path.join(out_path, "hist_log_passed_SGBs%s.png" % name))
        plt.close('all')

        n_SGBs = pandas.Series({x.split("|")[-1]: x for x in SGBs})
        for i, p in enumerate(df_passed.index[:10]):
            plt.scatter(lfold_df[df_passed.loc[p].oligo], MB_df[n_SGBs.loc[df_passed.loc[p].SGB]])
            plt.xlabel(df_passed.loc[p].oligo)
            plt.ylabel("\nf".join(n_SGBs.loc[df_passed.loc[p].SGB].split("|f")))
            plt.title("Corr of %g (pval %g)" % (df_passed.loc[p].spearman_corr, df_passed.loc[p].spearman_pval))
            plt.tight_layout()
            plt.savefig(os.path.join(out_path, "corr%s%d_%s_%s.png" % (name, i, df_passed.loc[p].oligo, df_passed.loc[p].SGB)))
            plt.close('all')

        plt.figure(figsize=(15, 5))
        corrs = df[['SGB', 'oligo', 'spearman_corr']].set_index(['SGB', 'oligo']).unstack()
        corrs.columns = corrs.columns.get_level_values(1)
        plt.imshow(corrs.values, cmap='plasma')
        plt.xlabel("oligos")
        plt.ylabel("SGBs")
        cbar = plt.colorbar()
        cbar.ax.set_ylabel("spearman correlation", rotation=-90, va="bottom")
        plt.title("Spearman correlation between bacterial abundance and oligo fold change")
        plt.savefig(os.path.join(out_path, "all%s_corrs.png" % name))
        plt.close('all')

    print()
