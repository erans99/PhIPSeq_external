import os
import sys
import time

import pandas
from scipy.stats import chisquare
from statsmodels.stats.multitest import multipletests

import PhIPSeq_external.config as config

base_path = config.ANALYSIS_PATH

if __name__ == '__main__':
    diag = "CD"
    MIN_APPEAR = 0.05
    NUM_TAKE = 50
    CV = 10

    cache_path = os.path.join(base_path, "Cache")
    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent_final.pkl"))
    out_path = os.path.join(base_path, "other")
    local_cache = os.path.join(out_path, "Cache")
    if not os.path.exists(local_cache):
        print("Make cache first")
        sys.exit()
    out_path = os.path.join(out_path, "%s_matched" % diag)
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    try:
        fold = pandas.read_pickle(os.path.join(local_cache, "matched_%s_fold_%d.pkl" % (diag, NUM_TAKE)))
        meta = pandas.read_pickle(os.path.join(local_cache, "matched_%s_meta_%d.pkl" % (diag, NUM_TAKE)))
        IL_fold = pandas.read_pickle(os.path.join(local_cache, "matchedIL_%s_fold_%d.pkl" % (diag, NUM_TAKE)))
        IL_meta = pandas.read_pickle(os.path.join(local_cache, "matchedIL_%s_meta_%d.pkl" % (diag, NUM_TAKE)))
    except:
        print("No cache exists")
        sys.exit()

    cols = pandas.read_csv(os.path.join(out_path, "single_on_%s_%d.csv" % (diag, NUM_TAKE)), index_col=0).index
    if os.path.exists(os.path.join(out_path, "predict_single.csv")):
        tmp_cols = pandas.read_csv(os.path.join(out_path, "predict_single.csv"), index_col=0)
        if tmp_cols.index != cols:
            print("Columns don;t match need to check")
            sys.exit()
        exclude = tmp_cols[tmp_cols.coagulation_related == 1].index
        ex_cols = list(set(cols).difference(exclude))
        print(len(cols), len(ex_cols), len(exclude))
    else:
        exclude = []

    print("Checking only on %d old 5%% columns" % len(cols))

    res = {}
    res_cols = ["num_passed_IL_healthy", "num_passed_NL_healthy", "num_passed_NL_CD",
                "chisq_ILh_NLh", "chisq_ILh_NLcd", "chisq_NLh_NLcd", "is_ex"]
    for c in cols:
        if (len(res) % 20) == 0:
            print("At %d of %d" % (len(res), len(cols)), time.ctime())
        try:
            res[c] = [(IL_fold[c].values > 0).sum(),
                      (fold[meta.StudyTypeID == 32][c].values > 0).sum(),
                      (fold[meta.StudyTypeID == 33][c].values > 0).sum()]
            for i in range(3):
                for j in range(i):
                    f_obs = [res[c][i], NUM_TAKE - res[c][i], res[c][j], NUM_TAKE - res[c][j]]
                    f_exp = []
                    for k in range(4):
                        f_exp.append((f_obs[k] + f_obs[k ^ 1]) * (f_obs[k] + f_obs[k ^ 2]) / sum(f_obs))
                    # if 0 in f_exp:
                    #     print()
                    res[c] += [chisquare(f_obs, f_exp, 1)[1]]
            if c in exclude:
                res[c].append(True)
            else:
                res[c].append(False)
        except:
            print("WTF %s" % c, fold[c].shape)
    res = pandas.DataFrame(res, index=res_cols).T
    for c in ["chisq_ILh_NLh", "chisq_ILh_NLcd", "chisq_NLh_NLcd"]:
        res['FDR_' + c] = multipletests(list(res[c].values), method='fdr_bh')[0]
    res['is_flag'] = df_info.loc[res.index].is_bac_flagella.values
    res['prot_name'] = df_info.loc[res.index]['full name'].values
    res.to_csv(os.path.join(out_path, 'compare_IL_NL_%d_%s.csv' % (NUM_TAKE, diag)))
