import string
import seaborn as sns
from matplotlib import pyplot as plt

from PhIPSeq_external.Analysis_allergens_and_IEDB.prediction_helpers import get_oligos, read_metadata, configure_x_y, \
    tune_hyper_parameters, plot_correlations, draw_auc, plot_shap


def main(tune_params=False):
    x = get_oligos()
    meta = read_metadata()
    meta = meta[['yob', 'gender']]
    y = meta['gender']
    meta.drop(columns=['gender'], inplace=True)
    x, y = configure_x_y(x, y, meta, is_classifier=True)
    if tune_params:
        params = tune_hyper_parameters(x, y, is_classifier=False)
    else:
        params = {'n_estimators': 185, 'max_depth': 4, 'min_samples_leaf': 36}

    fig = plt.figure(figsize=(20, 5))
    spec = fig.add_gridspec(1, 3)
    color = sns.color_palette()[4]

    # Create sub figure a
    ax = fig.add_subplot(spec[0])
    plot_correlations(x, y, 'gender', 'Number of male individuals in whom\na peptide is significantly bound',
                      'Number of female individuals in whom\na peptide is significantly bound', ax, color=color,
                      overwrite=True)
    ax.text(-0.15, 1.0, string.ascii_lowercase[0], transform=ax.transAxes, size=14, weight='bold')

    # Create sub figure b
    ax = fig.add_subplot(spec[1])
    ax = draw_auc(x, y, ax=ax, num_iterations=100, color=color, **params)
    ax.text(-0.15, 1.0, string.ascii_lowercase[1], transform=ax.transAxes, size=14, weight='bold')

    # Create sub figure c
    ax = plot_shap(x, y, True, params, 'gender', specs=spec[2], fig=fig, overwrite=True)
    ax.text(-0.15, 1.0, string.ascii_lowercase[2], transform=ax.transAxes, size=14, weight='bold')


if __name__ == "__main__":
    main()
    plt.show()
