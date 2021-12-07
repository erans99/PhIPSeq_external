import os
import string
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import fisher_exact
from sklearn.metrics import r2_score
from statsmodels.stats.multitest import fdrcorrection

from PhIPSeq_external.Analysis_allergens_and_IEDB.config import base_path, out_path
from PhIPSeq_external.Analysis_allergens_and_IEDB.prediction_helpers import get_oligos, configure_x_y, \
    tune_hyper_parameters, predict_with_cross_validation, plot_shap, read_metadata


def compute_age_correlations(x, y, overwrite=True):
    csv_out_path = os.path.join(out_path, 'age_correlations_data.csv')
    if overwrite or not os.path.exists(csv_out_path):
        os.makedirs(out_path, exist_ok=True)
        bottom_quintile = y.quantile(0.2)
        top_quintile = y.quantile(0.8)
        bottom_label = f"Percent of youngest quintile (<{int(bottom_quintile)}\nin whom a peptide is significantly bound"
        top_label = f"Percent of oldest quintile (>{int(top_quintile)}\nin whom a peptide is significantly bound"
        age_group = y.apply(
            lambda v: bottom_label if v < bottom_quintile else (top_label if v > top_quintile else np.nan))
        oligos = x.drop(columns=['gender']).gt(1)
        oligos['age_group'] = age_group
        oligos.dropna(inplace=True)

        # Perform Fisher's exact test to identify significant oligos
        summed_outcome = age_group.value_counts()
        summed_df = oligos.groupby('age_group').sum().T
        summed_df.rename(columns={k: f"{k}_1" for k in summed_df.columns}, inplace=True)
        for col in summed_outcome.index:
            summed_df[f"{col}_0"] = summed_outcome.loc[col] - summed_df[f"{col}_1"]
        summed_df['oddsratio'] = summed_df.apply(
            lambda row: fisher_exact([[row[bottom_label + '_0'], row[bottom_label + '_1']],
                                      [row[top_label + '_0'], row[top_label + '_1']]])[0], axis=1)
        summed_df['p-value'] = summed_df.apply(
            lambda row: fisher_exact([[row[bottom_label + '_0'], row[bottom_label + '_1']],
                                      [row[top_label + '_0'], row[top_label + '_1']]])[1], axis=1)
        summed_df['passed_fdr'], summed_df['fdr_corrected_p_value'] = fdrcorrection(summed_df['p-value'])

        oligos = oligos.groupby('age_group').mean().T * 100.0
        oligos = oligos.merge(summed_df[['oddsratio', 'p-value', 'passed_fdr', 'fdr_corrected_p_value']],
                              left_index=True, right_index=True, how='left')
        df_info = pd.read_csv(os.path.join(base_path, "library_contents.csv"), index_col=0, low_memory=False)[
            ['nuc_seq', 'full name']]
        oligos = oligos.merge(df_info, left_index=True, right_index=True, how='left')
        oligos.to_csv(csv_out_path)
    return pd.read_csv(csv_out_path, index_col=0)


def make_age_correlation_subplot(x, y, ax=None):
    if ax is None:
        ax = plt.subplot()
    oligos = compute_age_correlations(x, y)
    bottom_label = [col for col in oligos.columns if '(<' in col][0]
    top_label = [col for col in oligos.columns if '(>' in col][0]
    sns.scatterplot(data=oligos, x=bottom_label, y=top_label, hue='passed_fdr',
                    palette=[sns.color_palette()[0], 'black'], legend=True, ax=ax)
    legend_handles = ax.get_legend_handles_labels()
    ax.legend(
        [legend_handles[0][legend_handles[1].index('True')], legend_handles[0][1 - legend_handles[1].index('True')]],
        ['Passed FDR', 'Did not pass FDR'], title="Fisher's exact text")
    # TODO: add annotations
    return ax


def make_age_prediction_figure(x, y, params, ax=None):
    if ax is None:
        ax = plt.subplot()
    prediction_path = os.path.join(out_path, 'age_prediction_results.csv')
    if not os.path.exists(prediction_path):
        predictions_results = predict_with_cross_validation(x, y, is_classifier=False, return_predictions=True,
                                                            **params)
        predictions_results.to_csv(prediction_path)
    predictions_results = pd.read_csv(prediction_path, index_col=0)
    sns.scatterplot(data=predictions_results, x='y', y='y_hat', alpha=0.8,
                    label=f'$R^2={r2_score(predictions_results["y"], predictions_results["y_hat"]):.2f}$', ax=ax)
    moving_average = predictions_results.sort_values(by="y").rolling(window=50).mean().dropna()
    ax.plot(moving_average["y"], moving_average["y_hat"], label='Moving average', color='r', linestyle='-')
    ax.legend()
    ax.set_xlabel("Age")
    ax.set_ylabel("Predicted age")
    return ax


def main(tune_params=False):
    x = get_oligos()
    meta = read_metadata()
    meta = meta[['age', 'gender']]
    y = meta['age']
    meta.drop(columns=['age'], inplace=True)
    x, y = configure_x_y(x, y, meta, is_classifier=False)
    if tune_params:
        params = tune_hyper_parameters(x, y, is_classifier=False)
    else:
        params = {'n_estimators': 178, 'max_depth': 2, 'min_samples_leaf': 66}

    fig = plt.figure(figsize=(20, 5))
    spec = fig.add_gridspec(1, 3)

    # Create sub figure a
    ax = fig.add_subplot(spec[0])
    make_age_correlation_subplot(x, y, ax)
    ax.text(-0.15, 1.0, string.ascii_lowercase[0], transform=ax.transAxes, size=14, weight='bold')

    # Create sub figure b
    ax = fig.add_subplot(spec[1])
    make_age_prediction_figure(x, y, params, ax)
    ax.text(-0.15, 1.0, string.ascii_lowercase[1], transform=ax.transAxes, size=14, weight='bold')

    # Create sub figure c
    ax = plot_shap(x, y, False, params, 'age', specs=spec[2], fig=fig, overwrite=True)
    ax.text(-0.15, 1.0, string.ascii_lowercase[2], transform=ax.transAxes, size=14, weight='bold')


if __name__ == "__main__":
    main()
    plt.show()
