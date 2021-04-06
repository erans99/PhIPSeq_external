import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats
import shap
import xgboost as xgb
from matplotlib.lines import Line2D
from sklearn.metrics import roc_curve, auc

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

lib = "agilent"
MIN_OLIS = 200
NUM_FOLDS = 10
MIN_SAMPS_PROP = 0.02
SHIFT = 100

NUM_ROUNDS = 100

params = {'colsample_bylevel': 0.075, 'max_depth': 6,
          'learning_rate': 0.0025, 'n_estimators': 4000, 'subsample': 0.6, 'min_child_weight': 20}

NUM_THREADS = 16
RUN_FIT = True

CHECK = None
EXT = ['age', 'gender']
OLIGO_STAT = [True]
INCLUDE_STAT = ['bacterial', 'microbiome', 'PNP', 'VFDB']
TYPE_STAT = ['fold']


def check_shap(pheno, sbj_inf, cache_path, OLIGO, TYPE, INCLUDE, ext=pandas.DataFrame()):
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
            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                     "Olis_bac_%s" % TYPE)
        elif INCLUDE == 'microbiome':
            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                     "Olis_microbiome_%s" % TYPE)
        elif INCLUDE == 'PNP':
            df_info = df_info[df_info.is_PNP]
            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                     "Olis_PNP_%s" % TYPE)
        elif INCLUDE == 'VFDB':
            df_info = df_info[df_info.is_toxin]
            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                     "Olis_VFDB_%s" % TYPE)
        else:
            print("No such include %s" % INCLUDE)
            sys.exit()
    else:
        print("Working on prots %s %s" % (TYPE, INCLUDE))
        df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_%s_prot.pkl" % lib))
        if INCLUDE == 'bacterial':
            df_info = df_info[~df_info.is_IEDB_or_cntrl]
            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                     "Prots_bac_%s" % TYPE)
        elif INCLUDE == 'microbiome':
            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                     "Prots_microbiome_%s" % TYPE)
        elif INCLUDE == 'PNP':
            df_info = df_info[df_info.is_PNP]
            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                     "Prots_PNP_%s" % TYPE)
        elif INCLUDE == 'VFDB':
            df_info = df_info[df_info.is_toxin]
            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                     "Prots_VFDB_%s" % TYPE)
        else:
            print("No such include %s" % INCLUDE)
            sys.exit()
    out_path = pred_path

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
            exist_df = pandas.read_pickle(os.path.join(cache_path, "%s_protein_exist_allsrc.pkl" %
                                                       lib)).loc[metadata.index]
            if TYPE == 'fold':
                fold_df = pandas.read_pickle(os.path.join(cache_path, "%s_protein_fold_allsrc.pkl" % lib))
                fold_df = fold_df.loc[metadata.index].copy()
                fold_df.index = metadata.index
                fold_df.fillna(1, inplace=True)
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
    inds = exist_df.columns.intersection(df_info.index)
    exist_df = exist_df[inds]
    fold_df = fold_df[inds]

    num_samples_per_oli = exist_df.sum(0)
    olis = num_samples_per_oli[num_samples_per_oli > (MIN_SAMPS_PROP * len(exist_df))].index
    fold_olis = fold_df[olis]
    fold_olis.index = metadata.RegistrationCode

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    locs = numpy.random.permutation(locs)
    fold = 0
    test_set = locs[int((fold / NUM_FOLDS) * len(locs)):int(((fold + 1) / NUM_FOLDS) * len(locs))]
    train_set = list(set(locs).difference(test_set))

    if len(set(sbj_inf.loc[locs].values)) > 2:
        model = xgb.XGBRegressor(nthread=NUM_THREADS, objective='reg:squarederror', **params)
    else:
        model = xgb.XGBClassifier(nthread=NUM_THREADS, **params)

    if len(ext.columns) == 0:
        model.fit(fold_olis.loc[train_set].values, sbj_inf.loc[train_set].values)
    else:
        model.fit(numpy.concatenate((fold_olis.loc[train_set].values, ext.loc[train_set].values), axis=1),
                  sbj_inf.loc[train_set].values)

    s = shap.TreeExplainer(model)
    if len(ext.columns) == 0:
        sv = s.shap_values(fold_olis.loc[test_set].values)
    else:
        sv = s.shap_values(numpy.concatenate((fold_olis.loc[test_set].values, ext.loc[test_set].values), axis=1))
    top_shap = [
        numpy.concatenate((fold_olis.columns, ext.columns))[numpy.argsort(numpy.sum(numpy.abs(sv), axis=0))[::-1]],
        numpy.sort(numpy.sum(numpy.abs(sv), axis=0))[::-1]]
    if OLIGO:
        top_shap = pandas.DataFrame(top_shap, index=['oligo', 'sum_abs_shap']).T.set_index('oligo')
    else:
        top_shap = pandas.DataFrame(top_shap, index=['prot', 'sum_abs_shap']).T.set_index('prot')
    top_shap['shap_spearman'] = None
    top_shap['pheno_spearman'] = None
    for i in range(len(top_shap)):
        oli = top_shap.iloc[i].name
        if oli in ext.columns:
            continue
        top_shap.loc[oli, 'shap_spearman'] = scipy.stats.spearmanr(sv[:, i], fold_olis.loc[test_set, oli])[0]
        top_shap.loc[oli, 'pheno_spearman'] = scipy.stats.spearmanr(sbj_inf.loc[test_set].values,
                                                                    fold_olis.loc[test_set, oli])[0]

    top_shap = top_shap.merge(df_info.loc[top_shap.index.intersection(df_info.index)], 'left',
                              left_index=True, right_index=True)

    if not OLIGO:
        fold_olis.columns = df_info.loc[fold_olis.columns].prot_ID.values
    if len(ext.columns) == 0:
        top_shap.to_csv(os.path.join(out_path, "shap_%s.csv" % pheno))
        shap.summary_plot(sv, fold_olis.loc[test_set], show=False)
    else:
        top_shap.to_csv(os.path.join(out_path, "shap_ext_%s.csv" % pheno))
        shap.summary_plot(sv, pandas.concat([fold_olis.loc[test_set], ext.loc[test_set]], axis=1), show=False)

    if OLIGO:
        plt.title("Impact on %s of oligos and %s (%g train, %g test)" % (pheno, list(ext.columns),
                                                                         ((NUM_FOLDS - 1) / NUM_FOLDS),
                                                                         (1 / NUM_FOLDS)))
    else:
        plt.title("Impact on %s of prots and %s (%g train, %g test)" % (pheno, list(ext.columns),
                                                                        ((NUM_FOLDS - 1) / NUM_FOLDS), (1 / NUM_FOLDS)))

    plt.tight_layout()
    if len(ext.columns) == 0:
        plt.savefig(os.path.join(out_path, "shap_%s.png" % pheno))
    else:
        plt.savefig(os.path.join(out_path, "shap_ext_%s.png" % pheno))
    plt.close('all')


def fit_pheno(pheno, sbj_inf, inds, cache_path, pred_path, OLIGO, TYPE):
    print(pred_path)
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
    olis = num_samples_per_oli[num_samples_per_oli > (MIN_SAMPS_PROP * len(exist_df))].index
    fold_olis = fold_df[olis]
    fold_olis.index = metadata.RegistrationCode

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    locs = numpy.random.permutation(locs)
    cv_pred = []
    for fold in range(NUM_FOLDS):
        test_set = locs[int((fold / NUM_FOLDS) * len(locs)):int(((fold + 1) / NUM_FOLDS) * len(locs))]
        train_set = list(set(locs).difference(test_set))

        if len(set(sbj_inf.loc[locs].values)) > 2:
            model = xgb.XGBRegressor(nthread=NUM_THREADS,  objective='reg:squarederror', **params)
        else:
            model = xgb.XGBClassifier(nthread=NUM_THREADS, **params)
        model.fit(fold_olis.loc[train_set].values, sbj_inf.loc[train_set].values)
        # model.save_model(os.path.join(pred_path, "model_%s.mdl" % pheno))

        if len(set(sbj_inf.loc[locs].values)) > 2:
            train_predict = model.predict(fold_olis.loc[train_set].values)
            test_predict = model.predict(fold_olis.loc[test_set].values)

            train_r = scipy.stats.pearsonr(sbj_inf.loc[train_set].values, train_predict)
            test_r = scipy.stats.pearsonr(sbj_inf.loc[test_set].values, test_predict)
        else:
            prob = model.predict_proba(fold_olis.loc[train_set].values)
            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[train_set].values), 1 - numpy.array(list(prob[:, 0])))
            train_r = auc(fpr, tpr)
            prob = model.predict_proba(fold_olis.loc[test_set].values)
            test_predict = list(prob[:, 0])
            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[test_set].values), 1 - numpy.array(test_predict))
            test_r = auc(fpr, tpr)
        print("Fold #%d of %s: %s train, %s test" % (fold, pheno, train_r, test_r))
        cv_pred += list(test_predict)

    if len(set(sbj_inf.loc[locs].values)) > 2:
        test_r = scipy.stats.pearsonr(sbj_inf.loc[locs].values, cv_pred)
        test_R2 = comp_R2(sbj_inf.loc[locs].values, cv_pred)

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
            plt.title("%d fold CV of prediction from oligos on test set (%d values)\npearson corr %g p_val=%g" %
                      (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
        else:
            plt.title("%d fold CV of prediction from proteins on test set (%d values)\npearson corr %g p_val=%g" %
                      (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
        plt.xlabel("Actual %s" % pheno)
        plt.ylabel("Predicted %s" % pheno)
        plt.legend(handles=[mpatches.Patch(color="w", label=("R2 = %g" % test_R2)),
                            Line2D([0], [0], color='r', label="Moving Average")], frameon=False)
        if OLIGO:
            plt.savefig(os.path.join(pred_path, "oligos_%s_predict.png" % pheno))
        else:
            plt.savefig(os.path.join(pred_path, "prots_%s_predict.png" % pheno))
        plt.close('all')
        print("Final test of %s got R2 of %g" % (pheno, test_R2))
        out = "Final test of %s got R2 of %g %g %g" % (pheno, test_R2, test_r[0], test_r[1])
        open(os.path.join(pred_path, "test_%s.txt" % pheno), "w").write(out + "\n")
    else:
        fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[locs].values), 1 - numpy.array(cv_pred))
        test_r = auc(fpr, tpr)
        # best_th = threshold[pandas.Series(tpr-fpr).idxmax()]

        # tmp = pandas.DataFrame(index=locs)
        # tmp[pheno] = sbj_inf.loc[locs]
        # tmp['prob_pred'] = cv_pred
        # tmp['pred'] = [1*(x < best_th) for x in cv_pred]
        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        if OLIGO:
            plt.title("%d fold CV of prediction from oligos on test set (%d values)\nAUC %g" %
                      (NUM_FOLDS, len(locs), test_r))
        else:
            plt.title("%d fold CV of prediction from proteins on test set (%d values)\nAUC %g" %
                      (NUM_FOLDS, len(locs), test_r))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        if OLIGO:
            plt.savefig(os.path.join(pred_path, "oligos_%s_ROC.png" % pheno))
        else:
            plt.savefig(os.path.join(pred_path, "prots_%s_ROC.png" % pheno))
        plt.close("all")
        out = "Final test of %s got AUC of %g" % (pheno, test_r)
        open(os.path.join(pred_path, "test_%s.txt" % pheno), "w").write(out + "\n")

    print("Done", pred_path)


def create_pred(locs, input, output, pheno, pr=False):
    cv_pred = []
    for fold in range(NUM_FOLDS):
        test_set = locs[int((fold / NUM_FOLDS) * len(locs)):int(((fold + 1) / NUM_FOLDS) * len(locs))]
        train_set = list(set(locs).difference(test_set))

        if len(set(output.loc[locs].values)) > 2:
            model = xgb.XGBRegressor(nthread=NUM_THREADS,  objective='reg:squarederror', **params)
        else:
            model = xgb.XGBClassifier(nthread=NUM_THREADS, **params)

        model.fit(input.loc[train_set].values, output.loc[train_set].values)
        # model.save_model(os.path.join(out_path, "model_%s.mdl" % pheno))

        if len(set(output.loc[locs].values)) > 2:
            test_predict = model.predict(input.loc[test_set].values)
        else:
            prob = model.predict_proba(input.loc[test_set].values)
            test_predict = list(prob[:, 0])

        if pr:
            if len(set(output.loc[locs].values)) > 2:
                train_predict = model.predict(input.loc[train_set].values)
                train_r = scipy.stats.pearsonr(output.loc[train_set, pheno].values, train_predict)
                test_r = scipy.stats.pearsonr(output.loc[test_set, pheno].values, test_predict)
            else:
                prob = model.predict_proba(input.loc[train_set].values)
                train_predict = list(prob[:, 0])
                fpr, tpr, threshold = roc_curve(list(output.loc[train_set].values), 1 - numpy.array(train_predict))
                train_r = auc(fpr, tpr)
                fpr, tpr, threshold = roc_curve(list(output.loc[test_set].values), 1 - numpy.array(test_predict))
                test_r = auc(fpr, tpr)
            print("Full fold #%d of %s: %s train, %s test" % (fold, pheno, train_r, test_r))

        cv_pred += list(test_predict)
    return cv_pred


def fit_pheno_ext(pheno, sbj_inf, inds, cache_path, pred_path, ext, OLIGO, TYPE, plot=False, only_plot=False):
    if only_plot and not plot:
        print("Can't only plot if not ploting at all")
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
    olis = num_samples_per_oli[num_samples_per_oli > (MIN_SAMPS_PROP * len(exist_df))].index
    fold_olis = fold_df[olis]
    fold_olis.index = metadata.RegistrationCode

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    for c in ext.columns:
        locs = set(locs).intersection(ext[~numpy.isnan(ext[c])].index)
    locs = list(locs)

    out = ""
    pearsonr = [[], []]
    for i in range(NUM_ROUNDS):
        locs = numpy.random.permutation(locs)
        cv_pred = create_pred(locs, pandas.concat([fold_olis, ext], axis=1), sbj_inf, pheno)
        cv_ext_pred = create_pred(locs, ext, sbj_inf, pheno)

        if len(set(sbj_inf.loc[locs].values)) > 2:
            test_r = scipy.stats.pearsonr(sbj_inf.loc[locs].values, cv_pred)
            pearsonr[0].append(test_r[0])
            test_R2 = comp_R2(sbj_inf.loc[locs].values, cv_pred)
            test2_r = scipy.stats.pearsonr(sbj_inf.loc[locs].values, cv_ext_pred)
            pearsonr[1].append(test2_r[0])
        else:
            fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[locs].values), 1 - numpy.array(cv_pred))
            pearsonr[0].append(auc(fpr, tpr))
            test_R2 = numpy.nan
            fpr_ext, tpr_ext, threshold_ext = roc_curve(list(sbj_inf.loc[locs].values), 1 - numpy.array(cv_ext_pred))
            pearsonr[1].append(auc(fpr_ext, tpr_ext))

        if plot and (i == 0):
            if len(set(sbj_inf.loc[locs].values)) > 2:
                plt.scatter(sbj_inf.loc[locs].values, cv_pred)
                tmp = pandas.DataFrame(index=locs)
                tmp[pheno] = sbj_inf.loc[locs]
                tmp['pred'] = cv_pred
                tmp['pred2'] = cv_ext_pred
                tmp.sort_values(pheno, inplace=True)
                moving_avg = [[sum(tmp[pheno].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)],
                              [sum(tmp['pred'].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)]]
                plt.plot(moving_avg[0], moving_avg[1], c="r")
                moving_avg = [[sum(tmp[pheno].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)],
                              [sum(tmp['pred2'].values[i:i + SHIFT]) / SHIFT for i in range(len(locs) - SHIFT + 1)]]
                plt.plot(moving_avg[0], moving_avg[1], c="g")
                # ylim = plt.ylim()
                # plt.plot([ylim[0], ylim[1]], [ylim[0], ylim[1]], c="r")
                if OLIGO:
                    plt.title("%d fold CV of prediction from oligos on test set (%d values)\npearson corr %g p_val=%g" %
                              (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
                else:
                    plt.title(
                        "%d fold CV of prediction from proteins on test set (%d values)\npearson corr %g p_val=%g" %
                        (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
                plt.xlabel("Actual %s" % pheno)
                plt.ylabel("Predicted %s" % pheno)
                plt.legend(handles=[mpatches.Patch(color="w", label=("R2 = %g" % test_R2)),
                                    Line2D([0], [0], color='r', label="Moving Average"),
                                    Line2D([0], [0], color='g', label="Only %s" % list(ext.columns))], frameon=False)
                if OLIGO:
                    plt.savefig(os.path.join(pred_path, "oligos_%s_predict.png" % pheno))
                else:
                    plt.savefig(os.path.join(pred_path, "prots_%s_predict.png" % pheno))
                plt.close('all')
                print("Final test of %s got R2 of %g" % (pheno, test_R2))
            else:
                plt.plot(fpr, tpr, label="PhIP + %s: AUC %g" % (list(ext.columns), pearsonr[0][0]))
                plt.plot(fpr_ext, tpr_ext, label="only %s: AUC %g" % (list(ext.columns), pearsonr[1][0]))
                plt.plot([0, 1], [0, 1])
                if OLIGO:
                    plt.title("%d fold CV of prediction from oligos on test set (%d values)" %
                              (NUM_FOLDS, len(locs)))
                else:
                    plt.title("%d fold CV of prediction from proteins on test set (%d values)" %
                              (NUM_FOLDS, len(locs)))
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                if OLIGO:
                    plt.savefig(os.path.join(pred_path, "oligos_%s_ROC.png" % pheno))
                else:
                    plt.savefig(os.path.join(pred_path, "prots_%s_ROC.png" % pheno))
                plt.close("all")
                out = "Final test of %s got AUC of %g" % (pheno, pearsonr[0][0])
                open(os.path.join(pred_path, "test_%s.txt" % pheno), "w").write(out + "\n")
            if only_plot:
                return

        out += "Round %d\n" % i
        if len(set(sbj_inf.loc[locs].values)) > 2:
            out += "Final test of %s given [%s], got pearson %g %g\n" % (pheno, "_".join(list(ext.columns)),
                                                                         test_r[0], test_r[1])
            out += "Only ext test got pearson %g %g\n" % (test2_r[0], test2_r[1])
        else:
            out += "Final test of %s given [%s], got AUC %g\n" % (pheno, "_".join(list(ext.columns)), pearsonr[0][-1])
            out += "Only ext test got AUC %g\n" % pearsonr[1][-1]

    if len(set(sbj_inf.loc[locs].values)) > 2:
        out += "%s pearson 95%% CV %g - %g median %g\n" % (pheno, numpy.quantile(pearsonr[0], 0.05),
                                                           numpy.quantile(pearsonr[0], 0.95),
                                                           numpy.quantile(pearsonr[0], 0.5))
        out += "ext pearson 95%% CV %g - %g median %g\n" % (numpy.quantile(pearsonr[1], 0.05),
                                                            numpy.quantile(pearsonr[1], 0.95),
                                                            numpy.quantile(pearsonr[1], 0.5))
    else:
        out += "%s AUC 95%% CV %g - %g median %g\n" % (pheno, numpy.quantile(pearsonr[0], 0.05),
                                                       numpy.quantile(pearsonr[0], 0.95),
                                                       numpy.quantile(pearsonr[0], 0.5))
        out += "ext AUC 95%% CV %g - %g median %g\n" % (numpy.quantile(pearsonr[1], 0.05),
                                                        numpy.quantile(pearsonr[1], 0.95),
                                                        numpy.quantile(pearsonr[1], 0.5))

    out += str(pearsonr[0]) + "\n"
    out += str(pearsonr[1]) + "\n"
    print(out)
    open(os.path.join(pred_path, "test_ext", "test_ext_%s.txt" % pheno), "w").write(out)


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
                            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                     "Olis_bac_%s" % TYPE)
                        elif INCLUDE == 'microbiome':
                            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
                            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                     "Olis_microbiome_%s" % TYPE)
                        elif INCLUDE == 'PNP':
                            df_info = df_info[df_info.is_PNP]
                            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                     "Olis_PNP_%s" % TYPE)
                        elif INCLUDE == 'VFDB':
                            df_info = df_info[df_info.is_toxin]
                            pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                     "Olis_VFDB_%s" % TYPE)
                        else:
                            print("No such include %s" % INCLUDE)
                            sys.exit()
                    else:
                        print("Working on prots %s %s" % (TYPE, INCLUDE))
                        df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_%s_prot.pkl" % lib))
                        if INCLUDE == 'bacterial':
                            df_info = df_info[~df_info.is_IEDB_or_cntrl]
                            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                     "Prots_bac_%s" % TYPE)
                        elif INCLUDE == 'microbiome':
                            df_info = df_info[df_info.is_PNP | df_info.is_nonPNP_strains]
                            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                     "Prots_microbiome_%s" % TYPE)
                        elif INCLUDE == 'PNP':
                            df_info = df_info[df_info.is_PNP]
                            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                     "Prots_PNP_%s" % TYPE)
                        elif INCLUDE == 'VFDB':
                            df_info = df_info[df_info.is_toxin]
                            pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                     "Prots_VFDB_%s" % TYPE)
                        else:
                            print("No such include %s" % INCLUDE)
                            sys.exit()
                    if not os.path.exists(pred_path):
                        os.makedirs(pred_path)
                    if len(EXT) != 0:
                        if not os.path.exists(os.path.join(pred_path, "test_ext")):
                            os.makedirs(os.path.join(pred_path, "test_ext"))

                    print("Data len %d" % len(df_info))

                    if CHECK is None:
                        CHECK = sbj_inf.columns
                        if 'Date' in CHECK:
                            CHECK = CHECK.drop('Date')

                    for pheno in CHECK:
                        print("Sending %s" % pheno)
                        ext = EXT.copy()
                        if pheno in ext:
                            ext.pop(ext.index(pheno))
                        if len(EXT) == 0:
                            fit_pheno(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, OLIGO, TYPE)
                        else:
                            fit_pheno_ext(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, sbj_inf[ext],
                                          OLIGO, TYPE)
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
                        pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                 "Olis_bac_%s" % TYPE)
                    elif INCLUDE == 'microbiome':
                        pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                 "Olis_microbiome_%s" % TYPE)
                    elif INCLUDE == 'PNP':
                        pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                 "Olis_PNP_%s" % TYPE)
                    elif INCLUDE == 'VFDB':
                        pred_path = os.path.join(base_path, "predict", "Olis_%s" % TYPE,
                                                 "Olis_VFDB_%s" % TYPE)
                    else:
                        print("No such include %s" % INCLUDE)
                        sys.exit()
                else:
                    if INCLUDE == 'bacterial':
                        pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                 "Prots_bac_%s" % TYPE)
                    elif INCLUDE == 'microbiome':
                        pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                 "Prots_microbiome_%s" % TYPE)
                    elif INCLUDE == 'PNP':
                        pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                 "Prots_PNP_%s" % TYPE)
                    elif INCLUDE == 'VFDB':
                        pred_path = os.path.join(base_path, "predict", "Prots_%s" % TYPE,
                                                 "Prots_VFDB_%s" % TYPE)
                    else:
                        print("No such include %s" % INCLUDE)
                        sys.exit()

                res = {}
                if len(EXT) == 0:
                    pheno_2 = []
                    fs = glob.glob(os.path.join(pred_path, "test_*.txt"))
                    fs = list(filter(lambda x: '_ext_' not in x, fs))
                    for f in fs:
                        ps = open(f).read().split()
                        if ps[5] == 'R2':
                            res[os.path.basename(f)[len("test_"):-len(".txt")]] = [ps[3], ps[-3], ps[-2], ps[-1]]
                        else:
                            pheno_2.append(ps[3])
                            res[os.path.basename(f)[len("test_"):-len(".txt")]] = [ps[3], numpy.nan, ps[-1],
                                                                                   numpy.nan]
                    res = pandas.DataFrame(res).T
                    res.columns = ['pheno', 'R2', 'pearson_r', 'pearson_p']
                    res['pearson_p'] = res['pearson_p'].astype(float)
                    res.sort_values('pearson_p', inplace=True)
                    res.to_csv(os.path.join(pred_path, "res_test_bac_Oligos.csv"))

                    bonf = res[res.pearson_p < (0.05 / len(res))]

                    print("Checking %d bonferoni results + %d AUC results" % (len(bonf), len(pheno_2)))
                    for pheno in pheno_2:  # bonf['pheno'] + pheno_2:
                        check_shap(pheno, sbj_inf[pheno], cache_path, OLIGO, TYPE, INCLUDE)
                else:
                    fs = glob.glob(os.path.join(pred_path, "test_ext", "test_ext_*.txt"))
                    pearsonr = {}
                    for f in fs:
                        ls = open(f).read().split("\n")
                        p_val = []
                        for l in ls[1:-5:3]:
                            p_val.append(float(l.split()[-1]))
                        ps1 = ls[-5].split()
                        ps2 = ls[-4].split()
                        res[os.path.basename(f)[len("test_"):-len(".txt")]] = [ps1[0], float(ps1[4]),
                                                                               float(ps1[8]), float(ps1[6])]
                        res[os.path.basename(f)[len("test_"):-len(".txt")]] += [float(ps2[4]), float(ps2[8]),
                                                                                float(ps2[6])]
                        res[os.path.basename(f)[len("test_"):-len(".txt")]].append(numpy.quantile(p_val, 0.5))
                        pearsonr[ps1[0]] = []
                        pearsonr[ps1[0]].append(eval(ls[-3]))
                        pearsonr[ps1[0]].append(eval(ls[-2]))
                        mu, std = scipy.stats.norm.fit(pearsonr[ps1[0]][1])
                        res[os.path.basename(f)[len("test_"):-len(".txt")]] += [mu, std]

                        if not os.path.exists(os.path.join(pred_path, "hists_vs_ext")):
                            os.makedirs(os.path.join(pred_path, "hists_vs_ext"))
                        plt.figure()
                        plt.hist(pearsonr[ps1[0]][0], color="r", label="All features", alpha=0.5)
                        plt.hist(pearsonr[ps1[0]][1], color="b", label="Ext features", alpha=0.5)
                        plt.legend()
                        plt.title("Predicting %s, distribution of Pearson correlation" % ps1[0])
                        plt.savefig(os.path.join(pred_path, "hists_vs_ext", "hist_corrs_%s.png" % ps1[0]))
                        plt.close('all')

                    res = pandas.DataFrame(res).T
                    res.columns = ['pheno', 'pearson_r5', 'pearson_r50', 'pearson_r95', 'ext_pearson_r5',
                                   'ext_pearson_r50', 'ext_pearson_r95', 'med_p_val', 'ext_mu', 'ext_std']
                    res['added95'] = res['pearson_r5'] >= res['ext_pearson_r95']
                    res['added50'] = res['pearson_r5'] >= res['ext_pearson_r50']
                    res['min_added'] = (res['pearson_r5'] - res['ext_pearson_r95']) * res['added50']
                    res['ks_stat'] = None
                    res['log2_p_val_ks'] = None
                    for ind in res[res.med_p_val < (0.05 / len(res))].index:
                        pheno = res.loc[ind, 'pheno']
                        ks = scipy.stats.ks_2samp(pearsonr[pheno][0], pearsonr[pheno][1], 'less')
                        res.loc[ind, 'ks_stat'] = ks[0]
                        res.loc[ind, 'log2_p_val_ks'] = numpy.log2(ks[1])
                    res['log2_p_val'] = None
                    for ind in res[(res.med_p_val < (0.05 / len(res))) & (res['min_added'] > 0)].index:
                        res.loc[ind, 'log2_p_val'] = scipy.stats.norm.logsf(
                            (res.loc[ind, 'pearson_r50'] - res.loc[ind, 'ext_mu']) /
                            res.loc[ind, 'ext_std']) / numpy.log(2)

                    res.sort_values('log2_p_val_ks', inplace=True)
                    res.to_csv(os.path.join(pred_path, "res_test_ext.csv"))

                    print("Checking %d results" % len(res[res['log2_p_val_ks'] < 0]))
                    for pheno in res[res['log2_p_val_ks'] < 0]['pheno']:
                        ext = EXT.copy()
                        if pheno in ext:
                            ext.pop(ext.index(pheno))
                        check_shap(pheno, sbj_inf[pheno], cache_path, OLIGO, TYPE, INCLUDE, sbj_inf[ext])
    print("Done")
