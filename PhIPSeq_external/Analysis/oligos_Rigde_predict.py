import glob
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy
import pandas
import scipy.stats
from matplotlib.lines import Line2D
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import roc_curve, auc

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

lib = "agilent"
MIN_OLIS = 200
NUM_FOLDS = 10
MIN_SAMPS_PROP = 0.02
SHIFT = 100

NUM_ROUNDS = 100

RUN_FIT = True

CHECK = None
EXT = ['age', 'gender']
INCLUDE_STAT = ['bacterial', 'microbiome', 'PNP', 'VFDB']
TYPE_STAT = ['fold'] #, 'pval', 'exist']


def get_model(pheno, sbj_inf, cache_path, TYPE, INCLUDE, ext=pandas.DataFrame()):
    print("Working on oligos %s %s" % (TYPE, INCLUDE))
    df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent.pkl"))
    df_info['is_IEDB_or_cntrl'] = df_info.is_IEDB | df_info.is_pos_cntrl | df_info.is_neg_cntrl | \
                                  df_info.is_rand_cntrl
    if INCLUDE == 'bacterial':
        df_info = df_info[~df_info.is_IEDB_or_cntrl]
        pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                 "Olis_%s" % TYPE,
                                 "Olis_bac_%s" % TYPE)
    elif INCLUDE == 'microbiome':
        df_info = df_info[df_info.is_gut_microbiome | df_info.is_patho_strain |
                          df_info.is_IgA_coated_strain | df_info.is_probio_strain]
        pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                 "Olis_%s" % TYPE,
                                 "Olis_microbiome_%s" % TYPE)
    elif INCLUDE == 'PNP':
        df_info = df_info[df_info.is_gut_microbiome]
        pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                 "Olis_%s" % TYPE,
                                 "Olis_PNP_%s" % TYPE)
    elif INCLUDE == 'VFDB':
        df_info = df_info[df_info.is_VFDB]
        pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                 "Olis_%s" % TYPE,
                                 "Olis_VFDB_%s" % TYPE)
    else:
        print("No such include %s" % INCLUDE)
        sys.exit()
    out_path = pred_path

    try:
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        if TYPE == 'fold':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        elif TYPE == 'pval':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "pval_%s_above%d.pkl" % (lib, MIN_OLIS))).T
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

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    for c in ext.columns:
        locs = set(locs).intersection(ext[~numpy.isnan(ext[c])].index)
    locs = numpy.random.permutation(list(locs))

    if len(set(sbj_inf.loc[locs].values)) > 2:
        model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], normalize=True)
    else:
        model = SGDClassifier(loss='log')

    if len(ext.columns) == 0:
        model.fit(fold_olis.loc[locs].values, sbj_inf.loc[locs].values)
    else:
        model.fit(numpy.concatenate((fold_olis.loc[locs].values, ext.loc[locs].values), axis=1),
                  sbj_inf.loc[locs].values)
    # print(model.coef_)
    res = pandas.DataFrame(index=list(fold_olis.columns) + list(ext.columns))
    if type(model.coef_[0]) == numpy.ndarray:
        model.coef_ = model.coef_[0]
    res['coef'] = model.coef_
    res['abs_coef'] = numpy.abs(model.coef_)
    res.sort_values('abs_coef', ascending=False, inplace=True)
    if len(ext) == 0:
        res.merge(df_info, left_index=True, right_index=True).to_csv(os.path.join(pred_path, "ridge_coefs_%s.csv" %
                                                                                  pheno))
    else:
        res.merge(df_info, left_index=True, right_index=True).to_csv(os.path.join(pred_path, "ridge_coefs_ext_%s.csv" %
                                                                                  pheno))


def fit_pheno(pheno, sbj_inf, inds, cache_path, pred_path, TYPE):
    print(pred_path)
    try:
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        if TYPE == 'fold':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        elif TYPE == 'pval':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "pval_%s_above%d.pkl" % (lib, MIN_OLIS))).T
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

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    locs = numpy.random.permutation(locs)
    cv_pred = []
    for fold in range(NUM_FOLDS):
        test_set = locs[int((fold / NUM_FOLDS) * len(locs)):int(((fold + 1) / NUM_FOLDS) * len(locs))]
        train_set = list(set(locs).difference(test_set))

        if len(set(sbj_inf.loc[locs].values)) > 2:
            model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], normalize=True)
        else:
            model = SGDClassifier(loss='log')
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
        plt.title("%d fold CV of prediction from oligos on test set (%d values)\npearson corr %g p_val=%g" %
                  (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
        plt.xlabel("Actual %s" % pheno)
        plt.ylabel("Predicted %s" % pheno)
        plt.legend(handles=[mpatches.Patch(color="w", label=("R2 = %g" % test_R2)),
                            Line2D([0], [0], color='r', label="Moving Average")], frameon=False)
        plt.savefig(os.path.join(pred_path, "oligos_%s_predict.png" % pheno))
        plt.close('all')
        print("Final test of %s got R2 of %g" % (pheno, test_R2))
        out = "Final test of %s got R2 of %g %g %g" % (pheno, test_R2, test_r[0], test_r[1])
        open(os.path.join(pred_path, "test_%s.txt" % pheno), "w").write(out + "\n")
    else:
        fpr, tpr, threshold = roc_curve(list(sbj_inf.loc[locs].values), 1 - numpy.array(cv_pred))
        test_r = auc(fpr, tpr)

        plt.plot(fpr, tpr)
        plt.plot([0, 1], [0, 1])
        plt.title("%d fold CV of prediction from oligos on test set (%d values)\nAUC %g" %
                  (NUM_FOLDS, len(locs), test_r))
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.savefig(os.path.join(pred_path, "oligos_%s_ROC.png" % pheno))
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
            model = RidgeCV(alphas=[0.1, 1, 10, 100, 1000], normalize=True)
        else:
            model = SGDClassifier(loss='log')

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


def fit_pheno_ext(pheno, sbj_inf, inds, cache_path, pred_path, ext, TYPE, plot=False, only_plot=False):
    if only_plot and not plot:
        print("Can't only plot if not ploting at all")
    try:
        exist_df = pandas.read_pickle(os.path.join(cache_path, "exist_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        if TYPE == 'fold':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "log_fold_%s_above%d.pkl" % (lib, MIN_OLIS))).T
        elif TYPE == 'pval':
            fold_df = pandas.read_pickle(os.path.join(cache_path, "pval_%s_above%d.pkl" % (lib, MIN_OLIS))).T
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

    locs = sbj_inf[~numpy.isnan(sbj_inf)].index
    for c in ext.columns:
        locs = set(locs).intersection(ext[~numpy.isnan(ext[c])].index)

    out = ""
    pearsonr = [[], []]
    for i in range(NUM_ROUNDS):
        locs = numpy.random.permutation(list(locs))
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
                plt.title("%d fold CV of prediction from oligos on test set (%d values)\npearson corr %g p_val=%g" %
                          (NUM_FOLDS, len(locs), test_r[0], test_r[1]))
                plt.xlabel("Actual %s" % pheno)
                plt.ylabel("Predicted %s" % pheno)
                plt.legend(handles=[mpatches.Patch(color="w", label=("R2 = %g" % test_R2)),
                                    Line2D([0], [0], color='r', label="Moving Average"),
                                    Line2D([0], [0], color='g', label="Only %s" % list(ext.columns))], frameon=False)
                plt.savefig(os.path.join(pred_path, "oligos_%s_predict.png" % pheno))
                plt.close('all')
                print("Final test of %s got R2 of %g" % (pheno, test_R2))
            else:
                plt.plot(fpr, tpr, label="PhIP + %s: AUC %g" % (list(ext.columns), pearsonr[0][0]))
                plt.plot(fpr_ext, tpr_ext, label="only %s: AUC %g" % (list(ext.columns), pearsonr[1][0]))
                plt.plot([0, 1], [0, 1])
                plt.title("%d fold CV of prediction from oligos on test set (%d values)" %
                          (NUM_FOLDS, len(locs)))
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.legend()
                plt.savefig(os.path.join(pred_path, "oligos_%s_ROC.png" % pheno))
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
        sbj_inf = pandas.read_pickle(os.path.join(cache_path, "sbj_info_above%d.pkl" % MIN_OLIS)).T
        sbj_inf['pre_T2D'] = 1 * (sbj_inf.bt__hba1c >= 5.7)
    except:
        print("Make cache first")
        sys.exit()

    if RUN_FIT:
        for TYPE in TYPE_STAT:
            for INCLUDE in INCLUDE_STAT:
                print("Working on oligos %s %s" % (TYPE, INCLUDE))
                df_info = pandas.read_pickle(os.path.join(cache_path, "df_info_agilent.pkl"))
                df_info['is_IEDB_or_cntrl'] = df_info.is_IEDB | df_info.is_pos_cntrl | df_info.is_neg_cntrl | \
                                              df_info.is_rand_cntrl
                if INCLUDE == 'bacterial':
                    df_info = df_info[~df_info.is_IEDB_or_cntrl]
                    pred_path = os.path.join(base_path,
                                             "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                             "Olis_%s" % TYPE,
                                             "Olis_bac_%s" % TYPE)
                elif INCLUDE == 'microbiome':
                    df_info = df_info[df_info.is_gut_microbiome | df_info.is_patho_strain |
                                      df_info.is_IgA_coated_strain | df_info.is_probio_strain]
                    pred_path = os.path.join(base_path,
                                             "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                             "Olis_%s" % TYPE,
                                             "Olis_microbiome_%s" % TYPE)
                elif INCLUDE == 'PNP':
                    df_info = df_info[df_info.is_gut_microbiome]
                    pred_path = os.path.join(base_path,
                                             "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                             "Olis_%s" % TYPE,
                                             "Olis_PNP_%s" % TYPE)
                elif INCLUDE == 'VFDB':
                    df_info = df_info[df_info.is_VFDB]
                    pred_path = os.path.join(base_path,
                                             "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                             "Olis_%s" % TYPE,
                                             "Olis_VFDB_%s" % TYPE)
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
                        fit_pheno(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, TYPE)
                    else:
                        fit_pheno_ext(pheno, sbj_inf[pheno], df_info.index, cache_path, pred_path, sbj_inf[ext], TYPE)
    print("Done")

    for TYPE in TYPE_STAT:
        for INCLUDE in INCLUDE_STAT:
            if INCLUDE == 'bacterial':
                pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                         "Olis_%s" % TYPE,
                                         "Olis_bac_%s" % TYPE)
            elif INCLUDE == 'microbiome':
                pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                         "Olis_%s" % TYPE,
                                         "Olis_microbiome_%s" % TYPE)
            elif INCLUDE == 'PNP':
                pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                         "Olis_%s" % TYPE,
                                         "Olis_PNP_%s" % TYPE)
            elif INCLUDE == 'VFDB':
                pred_path = os.path.join(base_path,  "predict_Ridge%d" % int(MIN_SAMPS_PROP * 100),
                                         "Olis_%s" % TYPE,
                                         "Olis_VFDB_%s" % TYPE)
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
                for pheno in list(bonf['pheno']) + pheno_2:
                    get_model(pheno, sbj_inf[pheno], cache_path, TYPE, INCLUDE)
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
                                                                           float(ps1[8]),
                                                                           float(ps1[6])]
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

                res.sort_values(['log2_p_val', 'log2_p_val_ks'], inplace=True)
                res.to_csv(os.path.join(pred_path, "res_test_ext.csv"))

                print("Checking %d results" % len(res[res['log2_p_val_ks'] < numpy.log2(0.05)]))
                for pheno in res[res['log2_p_val_ks'] < numpy.log2(0.05)]['pheno']:
                    ext = EXT.copy()
                    if pheno in ext:
                        ext.pop(ext.index(pheno))
                    get_model(pheno, sbj_inf[pheno], cache_path, TYPE, INCLUDE, sbj_inf[ext])
    print("Done")
