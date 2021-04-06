import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats
import xgboost as xgb
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

import PhIPSeq_external.config as config

base_path = config.ANALYSIS_PATH

params_class = {'colsample_bylevel': 0.075, 'max_depth': 6,
                'learning_rate': 0.0025, 'n_estimators': 4000, 'subsample': 0.6, 'min_child_weight': 20}
params_pred = params_class.copy()
params_pred['objective'] = 'reg:squarederror'

lib = "agilent"
MIN_OLIS = 200
NUM_FOLDS = 10
MIN_OLI_SAMPS_PROP = 0.02
MIN_MB_SAMPS_PROP = 0.02
SHIFT = 100

NUM_ROUNDS = 10
NUM_THREADS = 16
RUN_FIT = True

OLIGO_STAT = [True]
INCLUDE_STAT = ['bacterial', 'microbiome', 'PNP', 'VFDB']
TYPE_STAT = ['fold']


def fit_pheno(pheno, sbj_inf, inds, cache_path, pred_path, OLIGO, TYPE, use_MB, use_PHIP, copy=0):
    if (not use_MB) and (not use_PHIP):
        print("No inputs. Need easier MB or PhIP")
        return
    print(pred_path)
    if use_PHIP:
        try:
            metadata = pandas.read_pickle(os.path.join(cache_path, "meta_%s_above%d.pkl" % (lib, MIN_OLIS)))
            if OLIGO:
                exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS)))
                if TYPE == 'fold':
                    fold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS)))
                elif TYPE == 'pval':
                    fold_df = pandas.read_pickle(os.path.join(cache_path, "pval_%s_above%d.pkl" % (lib, MIN_OLIS)))
                elif TYPE == 'exist':
                    fold_df = exist_df
                else:
                    print("WTF to type %s" % TYPE)
                    return
            else:
                exist_df = pandas.read_pickle(os.path.join(cache_path, "%s_protein_exist_allsrc.pkl" % lib)).loc[
                    metadata.index]
                if TYPE == 'fold':
                    fold_df = pandas.read_pickle(os.path.join(cache_path, "%s_protein_fold_allsrc.pkl" % lib))
                    fold_df = fold_df.loc[metadata.index].copy()
                    fold_df.index = metadata.index
                    fold_df.fillna(1, inplace=True)
                    fold_df[fold_df < 1.] = 1.
                    fold_df[fold_df < 1.] = 1.
                    fold_df = numpy.log(fold_df)
                elif TYPE == 'num_exist':
                    fold_df = pandas.read_pickle(os.path.join(cache_path, "%s_protein_num_exist_allsrc.pkl" %
                                                              lib)).loc[metadata.index]
                elif TYPE == 'exist':
                    fold_df = exist_df
                else:
                    print("WTF to type %s" % TYPE)
                    return
        except:
            print("Make cache first")
            sys.exit()
        inds = exist_df.columns.intersection(inds)
        exist_df = exist_df[inds]
        fold_df = fold_df[inds]

        num_samples_per_oli = exist_df.sum(0)
        olis = num_samples_per_oli[num_samples_per_oli > (MIN_OLI_SAMPS_PROP * len(exist_df))].index
        fold_olis = fold_df[olis]
        fold_olis.index = metadata.RegistrationCode

        inds = set(sbj_inf.index).intersection(set(fold_olis.index))
        fold_olis = fold_olis.reindex(inds)
        sbj_inf = sbj_inf.reindex(inds)

    else:
        print("Not using PhIP")

    if use_MB:
        MB_df = pandas.read_pickle(os.path.join(cache_path, "MB_%s_above%d.pkl" % (lib, MIN_OLIS)))
        MBinds = [x.split("_")[0] for x in MB_df.index]
        MB_df.index = MBinds
        cols = MB_df.columns[(MB_df > 0.0001).sum() > (MIN_MB_SAMPS_PROP * len(MB_df))]
        MB_df = MB_df[cols]

        MBinds = set(sbj_inf.index).intersection(set(MBinds))
        if use_PHIP:
            inds = MBinds.intersection(fold_olis.index)
            fold_olis = fold_olis.reindex(inds)
        else:
            inds = MBinds

        MB_df = MB_df.reindex(inds)
        sbj_inf = sbj_inf.reindex(inds)
    else:
        print("Not using MB")

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    locs = numpy.random.permutation(locs)
    cv_pred = []
    for fold in range(NUM_FOLDS):
        test_set = locs[int((fold / NUM_FOLDS) * len(locs)):int(((fold + 1) / NUM_FOLDS) * len(locs))]
        train_set = list(set(locs).difference(test_set))

        if len(set(sbj_inf.loc[locs].values)) > 2:
            model = xgb.XGBRegressor(nthread=NUM_THREADS, **params_pred)
        else:
            model = xgb.XGBClassifier(nthread=NUM_THREADS, **params_class)

        if use_PHIP & use_MB:
            inps = fold_olis.merge(MB_df, left_index=True, right_index=True)
            name = "MB & "
            fname = "MB_PH"
            if OLIGO:
                name += "PhIP oligos"
            else:
                name += "PhIP prots"
        elif use_PHIP:
            inps = fold_olis
            fname = "PH"
            if OLIGO:
                name = "PhIP oligos"
            else:
                name = "PhIP prots"
        else:
            inps = MB_df
            name = "MB"
            fname = "MB"

        model.fit(inps.loc[train_set].values, sbj_inf.loc[train_set].values)
        # model.save_model(os.path.join(pred_path, "model_%s.mdl" % pheno))

        if len(set(sbj_inf.loc[locs].values)) > 2:
            train_predict = model.predict(inps.loc[train_set].values)
            test_predict = model.predict(inps.loc[test_set].values)

            train_r = scipy.stats.pearsonr(sbj_inf.loc[train_set].values, train_predict)
            test_r = scipy.stats.pearsonr(sbj_inf.loc[test_set].values, test_predict)
        else:
            prob = model.predict_proba(inps.loc[train_set].values)
            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[train_set].values), 1 - numpy.array(list(prob[:, 0])))
            train_r = auc(fpr, tpr)
            prob = model.predict_proba(inps.loc[test_set].values)
            test_predict = list(prob[:, 0])
            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[test_set].values), 1 - numpy.array(test_predict))
            test_r = auc(fpr, tpr)
        print("Fold #%d of %s: %s train, %s test" % (fold, pheno, train_r, test_r))
        cv_pred += list(test_predict)

    pandas.Series(cv_pred, index=sbj_inf.loc[locs].index).to_csv(os.path.join(pred_path, "preds_%s_%s_%d.csv" %
                                                                              (fname, pheno, copy)))
    if len(set(sbj_inf.loc[locs].values)) > 2:
        test_r = scipy.stats.pearsonr(sbj_inf.loc[locs].values, cv_pred)
        test_R2 = comp_R2(sbj_inf.loc[locs].values, cv_pred)

        if copy == 0:
            plt.scatter(sbj_inf.loc[locs].values, cv_pred)
            tmp = pandas.DataFrame(index=locs)
            tmp[pheno] = sbj_inf.loc[locs]
            tmp['pred'] = cv_pred
            tmp.sort_values(pheno, inplace=True)
            moving_avg = [[sum(tmp[pheno].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)],
                          [sum(tmp['pred'].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)]]
            plt.plot(moving_avg[0], moving_avg[1], c="r")
            # ylim = plt.ylim()
            # plt.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], c="r")
            if OLIGO:
                plt.title("%d fold CV of prediction from %s on test set (%d values)\npearson corr %g p_val=%g" %
                          (NUM_FOLDS, name, len(locs), test_r[0], test_r[1]))
            else:
                plt.title("%d fold CV of prediction from %s on test set (%d values)\npearson corr %g p_val=%g" %
                          (NUM_FOLDS, name, len(locs), test_r[0], test_r[1]))
            plt.xlabel("Actual %s" % pheno)
            plt.ylabel("Predicted %s" % pheno)
            plt.legend(handles=[mpatches.Patch(color="w", label=("R2 = %g" % test_R2)),
                                Line2D([0], [0], color='r', label="Moving Average")], frameon=False)
            if OLIGO:
                plt.savefig(os.path.join(pred_path, "%s_%s_predict.png" % (fname, pheno)))
            else:
                plt.savefig(os.path.join(pred_path, "%s_%s_predict.png" % (fname, pheno)))
            plt.close('all')
        print("Final test of %s got R2 of %g" % (pheno, test_R2))
        out = "Final test of %s got R2 of %g %g %g" % (pheno, test_R2, test_r[0], test_r[1])
        open(os.path.join(pred_path, "test_%s_%s_%d.txt" % (fname, pheno, copy)), "w").write(out + "\n")
    else:
        fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[locs].values), 1 - numpy.array(cv_pred))
        test_r = auc(fpr, tpr)
        if copy == 0:
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1])
            if OLIGO:
                plt.title("%d fold CV of prediction from %s on test set (%d values)\nAUC %g" %
                          (NUM_FOLDS, name, len(locs), test_r))
            else:
                plt.title("%d fold CV of prediction from %s on test set (%d values)\nAUC %g" %
                          (NUM_FOLDS, name, len(locs), test_r))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            if OLIGO:
                plt.savefig(os.path.join(pred_path, "%s_%s_ROC.png" % (fname, pheno)))
            else:
                plt.savefig(os.path.join(pred_path, "%s_%s_ROC.png" % (fname, pheno)))
            plt.close("all")
        out = "Final test of %s got AUC of %g" % (pheno, test_r)
        open(os.path.join(pred_path, "test_%s_%s_%d.txt" % (fname, pheno, copy)), "w").write(out + "\n")

    print("Done copy %d of" % copy, pred_path)


def comp_R2(act, pred):
    m = sum(act) / len(act)
    SSres = sum((act - pred) ** 2)
    SStot = sum((act - m) ** 2)
    return 1 - SSres / SStot


if __name__ == '__main__':
    cache_path = os.path.join(base_path, "Cache")
    try:
        sbj_inf = pandas.read_pickle(os.path.join(cache_path, "sbj_info_above%d.pkl" % MIN_OLIS))
        sbj_inf['pre_T2D'] = 1 * (sbj_inf.bt__hba1c >= 5.7)
    except:
        print("Make cache first")
        sys.exit()

    if RUN_FIT:
        for OLIGO in OLIGO_STAT:
            for TYPE in TYPE_STAT:
                if OLIGO and (TYPE == 'num_exist'):
                    continue
                if (not OLIGO) and (TYPE == 'pval'):
                    continue
                for INCLUDE in INCLUDE_STAT:
                    if OLIGO:
                        print("Working on oligos %s %s" % (TYPE, INCLUDE))
                        df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent_final_allsrc.pkl"))
                        df_info.index = ["%s_%s" % (lib, x) for x in df_info.index]
                        cols = list(df_info.columns)
                        for c in ['end0_len15', 'hash0_len15', 'end1_len15', 'hash1_len15', 'end2_len14',
                                  'hash2_len14', 'end3_len15', 'hash3_len15', 'end4_len16', 'hash4_len16']:
                            cols.pop(cols.index(c))
                        df_info = df_info[cols]
                        if INCLUDE == 'bacterial':
                            df_info = df_info[~df_info.is_IEDB_or_cntrl]
                            pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                     "Olis_bac_%s" % TYPE)
                        elif INCLUDE == 'microbiome':
                            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
                            pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                     "Olis_microbiome_%s" % TYPE)
                        elif INCLUDE == 'PNP':
                            df_info = df_info[df_info.is_PNP]
                            pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                     "Olis_PNP_%s" % TYPE)
                        elif INCLUDE == 'VFDB':
                            df_info = df_info[df_info.is_toxin]
                            pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                     "Olis_VFDB_%s" % TYPE)
                        else:
                            print("No such include %s" % INCLUDE)
                            sys.exit()
                    else:
                        print("Working on prots %s %s" % (TYPE, INCLUDE))
                        df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_%s_prot.pkl" % lib))
                        if INCLUDE == 'bacterial':
                            df_info = df_info[~df_info.is_IEDB_or_cntrl]
                            pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                     "Prots_bac_%s" % TYPE)
                        elif INCLUDE == 'microbiome':
                            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
                            pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                     "Prots_microbiome_%s" % TYPE)
                        elif INCLUDE == 'PNP':
                            df_info = df_info[df_info.is_PNP]
                            pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                     "Prots_PNP_%s" % TYPE)
                        elif INCLUDE == 'VFDB':
                            df_info = df_info[df_info.is_toxin]
                            pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                     "Prots_VFDB_%s" % TYPE)
                        else:
                            print("No such include %s" % INCLUDE)
                            sys.exit()
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)

                    print("Data len %d" % len(df_info))

                    for pheno in ['age', 'gender']:
                        print("Sending %s" % pheno)
                        for i in range(NUM_ROUNDS):
                            if not os.path.exists(os.path.join(pred_path, "test_%s_%s_%d.txt" % ("MB_PH", pheno, i))):
                                fit_pheno(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, OLIGO, TYPE,
                                          True, True, i)
                            if not os.path.exists(os.path.join(pred_path, "test_%s_%s_%d.txt" % ("MB", pheno, i))):
                                fit_pheno(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, OLIGO, TYPE,
                                          True, False, i)
                            if not os.path.exists(os.path.join(pred_path, "test_%s_%s_%d.txt" % ("PH", pheno, i))):
                                fit_pheno(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, OLIGO, TYPE,
                                          False, True, i)
        print("Done")

    for OLIGO in OLIGO_STAT:
        for TYPE in TYPE_STAT:
            if OLIGO and (TYPE == 'num_exist'):
                continue
            if (not OLIGO) and (TYPE == 'pval'):
                continue
            for INCLUDE in INCLUDE_STAT:
                if OLIGO:
                    if INCLUDE == 'bacterial':
                        pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                 "Olis_bac_%s" % TYPE)
                    elif INCLUDE == 'microbiome':
                        pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                 "Olis_microbiome_%s" % TYPE)
                    elif INCLUDE == 'PNP':
                        pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                 "Olis_PNP_%s" % TYPE)
                    elif INCLUDE == 'VFDB':
                        pred_path = os.path.join(base_path, "predict_wMB", "Olis_%s" % TYPE,
                                                 "Olis_VFDB_%s" % TYPE)
                    else:
                        print("No such include %s" % INCLUDE)
                        sys.exit()
                else:
                    if INCLUDE == 'bacterial':
                        pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                 "Prots_bac_%s" % TYPE)
                    elif INCLUDE == 'microbiome':
                        pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                 "Prots_microbiome_%s" % TYPE)
                    elif INCLUDE == 'PNP':
                        pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                 "Prots_PNP_%s" % TYPE)
                    elif INCLUDE == 'VFDB':
                        pred_path = os.path.join(base_path, "predict_wMB", "Prots_%s" % TYPE,
                                                 "Prots_VFDB_%s" % TYPE)
                    else:
                        print("No such include %s" % INCLUDE)
                        sys.exit()

                res = {}
                for pheno in ['age', 'gender']:
                    ls = [[], [], []]
                    for i in range(NUM_ROUNDS):
                        df_MB = pandas.read_csv(os.path.join(pred_path, "preds_MB_%s_%d.csv" % (pheno, i)), index_col=0)
                        locs = df_MB.index
                        df_PH = pandas.read_csv(os.path.join(pred_path, "preds_PH_%s_%d.csv" % (pheno, i)), index_col=0)
                        df_PH = df_PH.loc[locs]
                        df_avg = ((df_MB + df_PH) / 2)['0']
                        locs = [str(x) for x in locs]
                        if len(set(sbj_inf.loc[locs, pheno].values)) > 2:
                            test_r = scipy.stats.pearsonr(sbj_inf.loc[locs, pheno].values, df_avg.values)
                            test_R2 = comp_R2(sbj_inf.loc[locs, pheno].values, df_avg.values)

                            print("Final test of %s got R2 of %g" % (pheno, test_R2), test_r)
                            ls[0].append(test_R2)
                            ls[1].append(test_r[0])
                            ls[2].append(test_r[1])
                        else:
                            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[locs, pheno].values),
                                                            1 - numpy.array(df_avg.values))
                            test_r = auc(fpr, tpr)
                            print("Final test of %s got AUC of %g" % (pheno, test_r))
                            ls[0].append(test_r)
                    ls[0] = numpy.array(ls[0])
                    if len(set(sbj_inf.loc[locs, pheno].values)) > 2:
                        ls[1] = numpy.array(ls[1])
                        ls[2] = numpy.array(ls[2])
                        res['%s_avg' % pheno] = [pheno, True, True, True, ls[0].mean(), ls[0].std(),
                                                 ls[1].mean(), ls[1].std(), ls[2].mean()]
                    else:
                        res['%s_avg' % pheno] = [pheno, True, True, True, ls[0].mean(), ls[0].std(),
                                                 numpy.nan, numpy.nan, numpy.nan]

                for name in ['MB_PH', 'MB', 'PH']:
                    for pheno in ['age', 'gender']:
                        ls = [[], [], []]
                        fs = glob.glob(os.path.join(pred_path, "test_%s_%s_*.txt" % (name, pheno)))
                        for f in fs:
                            ps = open(f).read().split()
                            if ps[5] == 'R2':
                                ls[0].append(float(ps[-3]))
                                ls[1].append(float(ps[-2]))
                                ls[2].append(float(ps[-1]))
                            else:
                                ls[0].append(float(ps[-1]))
                        ls[0] = numpy.array(ls[0])
                        if len(ls[1]) > 0:
                            ls[1] = numpy.array(ls[1])
                            ls[2] = numpy.array(ls[2])
                            res[pheno + "_" + name] = [pheno, 'MB' in name, 'PH' in name, False, ls[0].mean(),
                                                       ls[0].std(), ls[1].mean(), ls[1].std(), ls[2].mean()]
                        else:
                            res[pheno + "_" + name] = [pheno, 'MB' in name, 'PH' in name, False, ls[0].mean(),
                                                       ls[0].std(), numpy.nan, numpy.nan, numpy.nan]
                res = pandas.DataFrame(res).T
                res.columns = ['pheno', 'MB', 'PH', 'avg', 'R2/AUC mean', 'R2/AUC std', 'pearson_r mean',
                               'pearson_r std', 'pearson_p mean']
                res.sort_values(['pheno', 'MB', 'PH'], inplace=True)
                res.to_csv(os.path.join(pred_path, "all_res_test.csv"))

    print("Done")

    fs = glob.glob(os.path.join(base_path, "predict_wMB", "Olis_fold", "*", "all_res_test.csv"))
    dfs = {}
    for f in fs:
        dfs[f.split(os.sep)[-2].split("_")[1]] = pandas.read_csv(f, index_col=0)
    for pheno in ['age', 'gender']:
        plt.figure()
        plt.suptitle(pheno)
        for i, name in enumerate(dfs.keys()):
            plt.subplot("22%d" % (i + 1))
            tmp = dfs[name][dfs[name].pheno == pheno]
            tmp.index = ["predict_" + x.replace("_%s" % pheno, "").replace("%s_" % pheno, "").replace("_", "&")
                         for x in tmp.index]
            plt.barh(list(tmp.index), list(tmp['R2/AUC mean'].values), yerr=list(tmp['R2/AUC std'].values), height=0.4)
            if pheno != 'gender':
                plt.xlim([0, 0.6])
                plt.title("R2 of predictors from %s" % name)
            else:
                plt.xlim([0.5, 0.9])
                plt.title("AUC of predictors from %s" % name)
            plt.tight_layout()
        plt.savefig(os.path.join(base_path, "predict_wMB", "Olis_fold", "res_%s.png" % pheno))
    plt.show()
