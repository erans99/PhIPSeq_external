import os
import pandas
import numpy
import sys
import time
import matplotlib.pyplot as plt
import sklearn
from scipy.stats import ks_2samp, chisquare
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import r2_score, roc_curve, auc
import xgboost
from sklearn.linear_model import SGDClassifier
import seaborn as sns
from sklearn.decomposition import PCA

import PhIPSeq_external.config as config
base_path = config.ANALYSIS_PATH

NUM_TH = 16
MIN_APPEAR = 0.05

params = {'colsample_bytree': 0.1, 'max_depth': 4, 'learning_rate': 0.001,
          'n_estimators': 6000, 'subsample': 0.6, 'min_child_weight': 0.01}

predictor_class = xgboost.XGBClassifier(nthread=NUM_TH, **params)
pred_sgdc = SGDClassifier(loss='log')


def make_clasifier(predictor, in_data, mdl_file):
    df_in, df_out = in_data
    cv_predictor = sklearn.base.clone(predictor)
    cv_predictor.fit(df_in, df_out)
    cv_predictor.save_model(mdl_file)
    pandas.Series(df_in.columns).to_csv(mdl_file + "_cols.csv")


def pipeline_cross_val_x(predictor, in_data, CV=10, is_classifier=True, save_pred=None, plot=None, out_path=""):
    df_in, df_out = in_data

    results = pandas.DataFrame(index=df_out.index, columns=['y', 'y_hat', 'predicted_status'])
    results['y'] = df_out

    inds = list(df_in.sample(len(df_in)).index)

    jump = len(inds)/CV
    for i in range(CV):
        test_index = inds[int(i*jump): int((i+1)*jump)]
        train_index = inds[:int(i*jump)] +inds[int((i+1)*jump):]
        print("fold %d of %d" % (i, CV), time.ctime())
        X_train, X_test = df_in.loc[train_index], df_in.loc[test_index]
        y_train, y_test = df_out.loc[train_index], df_out.loc[test_index]
        cv_predictor = sklearn.base.clone(predictor)
        cv_predictor.fit(X_train, y_train)
        if is_classifier:
            prob = cv_predictor.predict_proba(X_test)
            results['y_hat'].loc[test_index] = numpy.ndarray.flatten(prob[:,cv_predictor.classes_ == 1])
            if numpy.ndarray.flatten(prob).max() > 1:
                print("impossible predicted probability")
            results['predicted_status'].loc[test_index] = cv_predictor.predict(X_test)
        else:
            pred = cv_predictor.predict(X_test)
            results['y_hat'].loc[test_index] = pred

    if save_pred is not None:
        results.to_csv(save_pred)

    if is_classifier:
        fpr, tpr, threshold = roc_curve(results['y'], results['y_hat'])
        res = auc(fpr, tpr)
        if plot is not None:
            plt.plot(fpr, tpr)
            plt.plot([0, 1], [0, 1])
            plt.title("%d-fold cross validation predicting disease status, AUC %.2g" % (CV, res))
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.savefig(os.path.join(out_path, "plots_%s_ROC.png" % plot.replace(" ", "_")))
            plt.close("all")

            res_pr = precision_recall_curve(results.y, results.y_hat)
            plt.plot(res_pr[1], res_pr[0])
            plt.hlines(0.5, 0, 1, linestyles="dashed")
            plt.title("PRC curve of leave one out predicting disease status")
            plt.xlabel("Recall (sensitivity)")
            plt.ylabel("Precision (PPV)")
            plt.savefig(os.path.join(out_path, "plots_%s_PRC.png" % plot.replace(" ", "_")))
            plt.close("all")
    else:
        res = r2_score(results['y'], results['y_hat'])
    # results['sample_id'] =
    return res, results['y_hat'].mean(), results['y_hat'].std(), results


def get_other_name(x, ser):
    tmp = ser[ser == x]
    if len(tmp) == 1:
        return tmp.index[0]
    else:
        print("WTF %s not found" % x)
        return None


def perform_dimensionality_reduction(out_path, existence_table, cohorts, dimensionality_reduction_class, data_type,
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
            pca_info[c] = [dimensionality_reduction_class.explained_variance_ratio_[i]]
            pca_info[c] += list(ks_2samp(transformed_table.loc[cohorts[cohorts].index][c].values,
                                         transformed_table.loc[cohorts[~cohorts].index][c].values, 'less'))
            pca_info[c] += list(ks_2samp(transformed_table.loc[cohorts[cohorts].index][c].values,
                                        transformed_table.loc[cohorts[~cohorts].index][c].values, 'greater'))
        pca_info = pandas.DataFrame(pca_info, index=['exp_var', 'ks_l_stat', 'ks_l_pval', 'ks_g_stat', 'ks_g_pval']).T
        pca_info.to_csv(os.path.join(out_path,  data_type + '_info.csv'))

    # Figure by status
    pca_info['min_ks'] = pca_info[['ks_g_stat', 'ks_g_pval']].min(1)
    pca_info.sort_values('min_ks', inplace=True)
    transformed_table.index = transformed_table.index.get_level_values(0)
    x = pca_info.index[0] #f'{column_prefix}1'
    y = pca_info.index[1] #f'{column_prefix}2'
    fig, ax = plt.subplots(ncols=1, nrows=1)
    sns.scatterplot(x=x, y=y, data=transformed_table, #transformed_table[x].values, y=transformed_table[y].values, 
                    hue=cohorts*1, ax=ax)
    if 'pca' in data_type:
        plt.xlabel('%s (explained variance %.2g%%)' % (x, 100 * pca_info.loc[x, 'exp_var']))
        plt.ylabel('%s (explained variance %.2g%%)' % (y, 100 * pca_info.loc[y, 'exp_var']))
    fig.savefig(os.path.join(out_path, f'{data_type}_by_status.png'))
    plt.close(fig)


def perform_pca(out_path, existence_table, cohorts, data_type, n_components=10, **kwargs):
    n_components = min(n_components, existence_table.shape[1])
    pca = PCA(n_components=n_components)
    perform_dimensionality_reduction(out_path, existence_table.fillna(0), cohorts, pca, data_type, 'PC', **kwargs)


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
    except:
        print("relevant cache not found")
        sys.exit()

    log = ""
    pr = ("For lib A, %d oligos passed for healthy %d for IBD" % ((fold[meta.StudyTypeID == 32] > 0).sum(1).mean(),
                                                                  (fold[meta.StudyTypeID == 33] > 0).sum(1).mean()))
    print(pr)
    log += pr + "\n"

    cols = (fold > 0).sum()
    cols = cols[cols > (MIN_APPEAR * len(fold))].index

    pr = ("%d oligos show up, %d in more then %g of cohort" % (fold.shape[1], len(cols), MIN_APPEAR))
    print(pr)
    log += pr + "\n"

    res = {}
    res_cols = ["ks_l_stat", "ks_l_pval", "ks_g_stat", "ks_g_pval", "num_passed_healthy", "num_passed_UC",
                "chisq_num_passed"]
    if not os.path.exists(os.path.join(out_path, 'single_on_%s_%d.csv' % (diag, NUM_TAKE))):
        for c in cols:
            if (len(res) % 20) == 0:
                print("At %d of %d" % (len(res), len(cols)), time.ctime())
            try:
                res[c] = list(ks_2samp(fold[meta.StudyTypeID == 32][c].values,
                                       fold[meta.StudyTypeID == 33][c].values, 'less'))
                res[c] += list(ks_2samp(fold[meta.StudyTypeID == 32][c].values,
                                        fold[meta.StudyTypeID == 33][c].values, 'greater'))
                res[c] += [(fold[meta.StudyTypeID == 32][c].values > 0).sum(),
                           (fold[meta.StudyTypeID == 33][c].values > 0).sum()]

                f_obs = [(fold[meta.StudyTypeID == 32][c] > 0).sum(),
                         (fold[meta.StudyTypeID == 32][c] == 0).sum(),
                         (fold[meta.StudyTypeID == 33][c] > 0).sum(),
                         (fold[meta.StudyTypeID == 33][c] == 0).sum()]
                f_exp = []
                for j in range(4):
                    f_exp.append((f_obs[j] + f_obs[j ^ 1]) * (f_obs[j] + f_obs[j ^ 2]) / sum(f_obs))
                res[c] += [chisquare(f_obs, f_exp, 1)[1]]
            except:
                print("WTF %s" % c, fold[c].shape)
        res = pandas.DataFrame(res, index=res_cols).T
        res['ks_pval'] = res[['ks_l_pval', 'ks_g_pval']].min(1)
        res.sort_values('ks_pval', inplace=True)
        other = list(res[['ks_l_pval', 'ks_g_pval']].max(1).values)
        FDR = multipletests(list(res['ks_pval'].values) + other, method='fdr_bh')[0]
        res['FDR_ks'] = FDR[:len(res)]
        res['FDR_chisq'] = multipletests(list(res['chisq_num_passed'].values), method='fdr_bh')[0]
        res['is_flag'] = df_info.loc[res.index].is_bac_flagella.values
        res['prot_name'] = df_info.loc[res.index]['full name'].values
        res.to_csv(os.path.join(out_path, 'single_on_%s_%d.csv' % (diag, NUM_TAKE)))
        pr = ("      oligos differentially expressed: %d by ks %d by chisq" % (len(res[res.FDR_ks]),
                                                                               len(res[res.FDR_chisq])))
        print(pr)
        log += pr + "\n"
    else:
        res = pandas.read_csv(os.path.join(out_path, 'single_on_%s_%d.csv' % (diag, NUM_TAKE)), index_col=0)

    if True:
        plt.scatter((fold[meta.StudyTypeID == 32] > 0).sum(0).values, (fold[meta.StudyTypeID == 33] > 0).sum(0).values,
                    label="All", color="cyan")
        inds = res[res.FDR_chisq].index
        plt.scatter((fold[meta.StudyTypeID == 32] > 0).sum(0)[inds].values, 
                    (fold[meta.StudyTypeID == 33] > 0).sum(0)[inds].values, label='Significantly Different', color="b")
        if os.path.exists(os.path.join(out_path, "excluse_%d_%s.csv" % (NUM_TAKE, diag))):
            exclude = pandas.read_csv(os.path.join(out_path, "excluse_%d_%s.csv" % (NUM_TAKE, diag)), index_col=0)
            exclude = exclude[exclude.coagulation_related == 1].index
            plt.scatter((fold[meta.StudyTypeID == 32] > 0).sum(0)[exclude].values, 
                        (fold[meta.StudyTypeID == 33] > 0).sum(0)[exclude].values, label='Exclude for Plasma vs. Serum',
                        color='k')
        plt.legend()
        plt.xlabel("Number of individual oligo appears in, of %d healthy" % NUM_TAKE)
        plt.ylabel("Number of individual oligo appears in, of %d %s patients" % (NUM_TAKE, diag))
        plt.savefig(os.path.join(out_path, "scatter_num_appear_in_%d.png" % NUM_TAKE))
        plt.close("all")

    if os.path.exists(os.path.join(out_path, "excluse_%d_%s.csv" % (NUM_TAKE, diag))):
        exclude = pandas.read_csv(os.path.join(out_path, "excluse_%d_%s.csv" % (NUM_TAKE, diag)), index_col=0)
        exclude = exclude[exclude.coagulation_related == 1].index
        ex_cols = list(set(cols).difference(exclude))
        print(len(cols), len(ex_cols), len(exclude))
    else:
        ex_cols = None

    if not os.path.exists(os.path.join(out_path, "preds_IBD_%s_%d.csv" % (diag, NUM_TAKE))):
        sc, mean, std, results = pipeline_cross_val_x(predictor_class, [fold[cols], (meta.StudyTypeID == 32)], CV=CV,
                                                      is_classifier=True, plot=("IBD_%d" % NUM_TAKE), out_path=out_path,
                                            save_pred=os.path.join(out_path, "preds_IBD_%s_%d.csv" % (diag, NUM_TAKE)))
        pr = ("On agilent lib oligos (appear >%g, %d oligos) %d-fold XGB classifier got AUC %g" % (MIN_APPEAR, len(cols),
                                                                                                   CV, sc))
        print(pr)
        log += pr + "\n\n"

    if ex_cols is not None:
        sc, mean, std, results = pipeline_cross_val_x(predictor_class, [fold[ex_cols], (meta.StudyTypeID == 32)],
                                                      CV=CV, is_classifier=True, plot="%s_ex_%d" % (diag, NUM_TAKE),
                                                      out_path=out_path,
                                        save_pred=os.path.join(out_path, "preds_%s_ex_%d.csv" % (diag, NUM_TAKE)))
        pr = ("On agilent lib oligos without serum related " +
              "(appear >%g, %d oligos) %d-fold XGB classifier got AUC %g" % (MIN_APPEAR, len(ex_cols), CV, sc))
        print(pr)
        log += pr + "\n\n"


    if not os.path.exists(os.path.join(out_path, "predictor_%s_%d.mdl" % (diag, NUM_TAKE))):
        make_clasifier(predictor_class, [fold[cols], (meta.StudyTypeID == 32)],
                       os.path.join(out_path, "predictor_%s_%d.mdl" % (diag, NUM_TAKE)))
    if not os.path.exists(os.path.join(out_path, "predictor_%s_ex_%d.mdl" % (diag, NUM_TAKE))):
        if ex_cols is not None:
            make_clasifier(predictor_class, [fold[ex_cols], (meta.StudyTypeID == 32)],
                           os.path.join(out_path, "predictor_%s_ex_%d.mdl" % (diag, NUM_TAKE)))

    if ex_cols is not None:
        name = os.path.join(out_path, 'sub_grps_prediction_%s_ex_%d' % (diag, NUM_TAKE))
    else:
        name = os.path.join(out_path, 'sub_grps_prediction_%s_%d' % (diag, NUM_TAKE))
    if not os.path.exists(name + ".csv"):
        res = {}
        for grp in ['all', 'VFDB', 'metagenomics', 'metagenomics flagellins']: #'VFDB flagellins',
            if grp == 'all':
                sub_cols = df_info[~df_info.is_IEDB_or_cntrl].index
            elif grp == 'VFDB':
                sub_cols = df_info[df_info.is_toxin].index
            elif grp == 'VFDB flaggelins':
                df_info['is_flagellin'] = [('flagell' in x.lower()) if type(x) is str else False
                                           for x in df_info.uniref_func.values]
                sub_cols = df_info[df_info.is_toxin&df_info.is_flagellin].index
            elif grp == 'metagenomics':
                sub_cols = df_info[df_info.is_PNP|df_info.is_nonPNP_strains].index
            elif grp == 'metagenomics flagellins':
                sub_cols = df_info[df_info.is_bac_flagella].index

            if ex_cols is not None:
                sub_cols = sub_cols.intersection(ex_cols)
            else:
                sub_cols = sub_cols.intersection(cols)
            print("%s working on %d oligos" % (grp, len(sub_cols)))
            scs = []
            for i in range(10):
                sc, _, _, _ = pipeline_cross_val_x(predictor_class, [fold[sub_cols], (meta.StudyTypeID == 32)],
                                                   CV=CV, is_classifier=True)
                scs.append(sc)
            res[grp] = [numpy.array(scs).mean(), numpy.array(scs).std(), scs]
            print("%s got %g +- %g" % (grp, res[grp][0], res[grp][1]), res[grp][2])
        res = pandas.DataFrame(res, index=['mean', 'std', 'vals']).T
        res.to_csv(name + ".csv")

        plt.bar(res.index, res['mean'].values, yerr=res['std'].values)
        plt.xticks(res.index, [x.replace(" ", "\n") for x in res.index])
        plt.ylim(0.5, 1)
        plt.title("AUC of predictions by different sub groups")
        plt.tight_layout()
        plt.savefig(name + '.png')
        plt.close('all')
    else:
        res = pandas.read_csv(name + '.csv', index_col=0)

    open(os.path.join(out_path, 'log_%s_%d.txt' % (diag, NUM_TAKE)), "w").write(log)
    print("Done")
