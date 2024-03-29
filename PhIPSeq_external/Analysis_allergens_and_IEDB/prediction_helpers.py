import os
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import shap
from matplotlib import pyplot as plt
from scipy.stats import fisher_exact
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve, auc, r2_score
from sklearn.model_selection import KFold, train_test_split
from statsmodels.stats.multitest import fdrcorrection

from PhIPSeq_external.Analysis_allergens_and_IEDB.config import base_path, out_path, MIN_OLIS


def get_oligos():
    meta = read_metadata()
    df = pd.read_csv(os.path.join(base_path, "fold_data.csv"), index_col=[0, 1], low_memory=False).loc[
        meta.index].unstack()
    df = df.T.reset_index(level=0, drop=True).T
    df = df.loc[:, df.min().ne(-1)].copy()

    # Filter to use only IEDB or allergens
    df_info = pd.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)
    df_info = df_info[df_info['num_copy'].eq(1)][['is_IEDB', 'is_allergens']]
    df_info = df_info[df_info.any(axis=1)].index
    df = df.loc[:, df.columns.isin(df_info)].copy()
    assert df.min().min() > 1
    df.fillna(1, inplace=True)
    df = np.log(df)
    return df


def configure_x_y(x, y, metadata=None, is_classifier=False):
    y = y.dropna()
    x = x.loc[x.index.intersection(y.index)]
    y = y.loc[x.index].copy()
    if is_classifier:
        y = y.astype(int)

    # filter oligos that appear in less than 5% of the population
    x = x.loc[:, x.gt(0).mean().gt(0.05)].copy()

    # Add year of birth and gender
    if metadata is None:
        metadata = read_metadata()[['yob', 'gender']]
    metadata.dropna(inplace=True)
    x = x.merge(metadata, left_index=True, right_index=True, how='inner')
    y = y.loc[x.index]
    assert x.shape[0] == y.shape[0]
    return x, y


def predict_with_cross_validation(x, y, is_classifier, return_predictions=False, random_state=1534321, **model_params):
    splitter = KFold(n_splits=10, random_state=random_state, shuffle=True)
    pred_ys = []
    true_ys = []
    ids = []
    for train_idx, test_idx in splitter.split(x):
        if is_classifier:
            clf = GradientBoostingClassifier(**model_params)
        else:
            clf = GradientBoostingRegressor(**model_params)
        train_x = x.iloc[train_idx]
        train_y = y.loc[train_x.index]
        test_x = x.iloc[test_idx]
        test_y = y.loc[test_x.index]
        clf.fit(train_x, train_y)
        if is_classifier:
            pred_y = clf.predict_proba(test_x)[:, 1]
        else:
            pred_y = clf.predict(test_x)
        pred_ys += pred_y.tolist()
        true_ys += test_y.values.tolist()
        ids += test_x.index.to_list()
    if return_predictions:
        return pd.DataFrame(data={'y': true_ys, 'y_hat': pred_ys}, index=ids)
    else:
        if is_classifier:
            return auc(*roc_curve(true_ys, pred_ys)[:2])
        else:
            return r2_score(true_ys, pred_ys)


def tune_single_hyper_parameter(x, y, is_classifier, parameter_name, parameter_range, **params):
    tmp_params = params.copy()
    results = {}
    for tuned_value in parameter_range:
        tmp_params[parameter_name] = tuned_value
        results[tuned_value] = predict_with_cross_validation(x, y, is_classifier=is_classifier, **tmp_params)
    results = pd.DataFrame(results)
    return results.idxmax()


def tune_hyper_parameters(x, y, is_classifier):
    params = {}
    params['n_estimators'] = tune_single_hyper_parameter(x, y, is_classifier, 'n_estimators', range(1, 401), **params)
    params['max_depth'] = tune_single_hyper_parameter(x, y, is_classifier, 'max_depth', range(1, 201), **params)
    params['min_samples_leaf'] = tune_single_hyper_parameter(x, y, is_classifier, 'min_samples_leaf', range(1, 201),
                                                             **params)
    return params


def draw_auc(x, y, ax=None, num_iterations=0, color='blue', **model_params):
    model_params['random_state'] = 46541357
    df = predict_with_cross_validation(x, y, is_classifier=True, return_predictions=True, **model_params)
    fpr, tpr, thresholds = roc_curve(df['y'], df['y_hat'])
    label = f"Predictor (AUC={auc(fpr, tpr):.3f}"
    if ax is None:
        ax = plt.subplot()
    if num_iterations > 0:
        auc_confidence_interval = []
        for _ in range(num_iterations):
            round_fprs, round_tprs, round_thresholds = roc_curve(
                *list(map(lambda item: item[1], train_test_split(df)[0].iteritems())))
            round_auc = auc(round_fprs, round_tprs)
            auc_confidence_interval.append(round_auc)
            ax.plot(round_fprs, round_tprs, color='grey', alpha=5 / num_iterations)
        label += f", std={np.std(auc_confidence_interval):.3f}"
    label += ")"

    ax.plot(fpr, tpr, color=color,
            label=label,
            lw=2,
            alpha=0.8, )
    ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
    ax.legend()
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    return ax


def plot_shap(x, y, is_classifier, params, target_name, specs=None, fig=None, overwrite=False):
    if fig is None:
        fig = plt.figure()
    if specs is None:
        specs = fig.add_gridspec(1, 1)
    external_ax = fig.add_subplot(specs)
    external_ax.axis('off')
    inner_spec = specs.subgridspec(1, 2, width_ratios=[1, 4], wspace=0)
    ax = fig.add_subplot(inner_spec[1])

    output_path_prefix = os.path.join(out_path, '_'.join([target_name, 'shap']))
    output_csv_path = output_path_prefix + '.csv'
    output_pickle_path = output_path_prefix + '.pkl'
    os.makedirs(os.path.dirname(output_path_prefix), exist_ok=True)
    if overwrite or not os.path.exists(output_pickle_path):
        model_class = GradientBoostingClassifier if is_classifier else GradientBoostingRegressor
        clf = model_class(**params).fit(x, y)
        explainer = shap.TreeExplainer(clf)
        shap_values = explainer(x)
        with open(output_pickle_path, 'wb') as f:
            pickle.dump(shap_values, f)
    with open(output_pickle_path, 'rb') as f:
        shap_values = pickle.load(f)

    # Save absolute Shapley values to table
    if overwrite or not os.path.exists(output_csv_path):
        shap_abs_values = pd.Series(data=np.abs(explainer.shap_values(x)).sum(axis=0),
                                    index=x.columns).sort_values(ascending=False)
        shap_abs_values = shap_abs_values.to_frame().rename(columns={0: ' Shapley absolute value'})
        df_info = pd.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)[
            ['nuc_seq', 'full name']]
        oligos_shap_values = pd.merge(shap_abs_values, df_info, left_index=True, right_index=True, how='left')
        oligos_shap_values.to_csv(output_csv_path)

    try:
        shap.plots.beeswarm(shap_values, show=False, sum_bottom_features=False, plot_size=None)
    except Exception as e:
        print("""You might not be using the right SHAP version. 
        Have a look at:
        github.com/kalkairis/shap""")
        shap.plots.beeswarm(shap_values, show=False, plot_size=None)

    ax.set_yticklabels(
        list(map(lambda label: plt.Text(*label.get_position(), label.get_text().split('_')[-1]),
                 ax.get_yticklabels())))
    ax.set_xlabel('SHAP value')
    return ax


def read_metadata():
    meta = pd.read_csv(os.path.join(base_path, "cohort.csv"), index_col=0, low_memory=False)
    meta = meta[meta.timepoint.eq(1) & meta.num_passed.ge(MIN_OLIS)]
    meta['age'] = pd.to_datetime(meta['Date']).dt.year - meta['yob']
    return meta


def plot_correlations(x, y, target_name, header_true, header_false, ax=None, overwrite=False,
                      color=sns.color_palette()[0]):
    if ax is None:
        ax = plt.subplot()
    output_path = os.path.join(out_path, '_'.join([target_name, 'correlations.csv']))
    if overwrite or not os.path.exists(output_path):
        df = x.filter(regex='^twist_', axis=1).gt(0).astype(int)
        df[target_name] = y.astype(int)
        summed_outcome = y.value_counts()
        df = df.groupby(target_name).sum().T
        fishers_test_df = df.rename(columns={k: f"{int(k)}_1" for k in df.columns})
        fishers_test_df['1_0'] = summed_outcome.loc[1] - fishers_test_df['1_1']
        fishers_test_df['0_0'] = summed_outcome.loc[0] - fishers_test_df['0_1']
        fishers_test_df = fishers_test_df.merge(pd.DataFrame(fishers_test_df.apply(
            lambda row: dict(
                zip(['oddsratio', 'p-value'], fisher_exact([[row['0_0'], row['0_1']], [row['1_0'], row['1_1']]]))),
            axis=1).to_dict()).T, left_index=True, right_index=True, how='left')
        fishers_test_df['passed_fdr'], fishers_test_df['fdr_corrected_p_value'] = fdrcorrection(
            fishers_test_df['p-value'])
        df = df.merge(fishers_test_df[['oddsratio', 'p-value', 'passed_fdr', 'fdr_corrected_p_value']],
                      left_index=True, right_index=True, how='left')
        # Rename headers
        df.rename(columns={True: header_true, False: header_false}, inplace=True)
        df_info = pd.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)[
            ['nuc_seq', 'full name']]
        df = df.merge(df_info, left_index=True, right_index=True, how='left')
        df.sort_values(by='fdr_corrected_p_value', inplace=True)
        df.to_csv(output_path)
    df = pd.read_csv(output_path, index_col=0)
    df.sort_values(by='fdr_corrected_p_value', ascending=False, inplace=True)
    ax = sns.scatterplot(data=df, y=header_true, x=header_false, ax=ax, legend=True, hue='passed_fdr',
                         palette=[color, 'black'], alpha=0.8)
    legend_handles = ax.get_legend_handles_labels()
    ax.legend(
        [legend_handles[0][legend_handles[1].index('True')], legend_handles[0][1 - legend_handles[1].index('True')]],
        ['Passed FDR', 'Did not pass FDR'])
    return ax
