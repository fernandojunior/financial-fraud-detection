import matplotlib.pyplot as plt
import seaborn as sns
import numpy
from catboost import Pool
import shap
# Copy value, not memory address
from copy import deepcopy

import utils
import features_engineering


def plot_heatmap(data_set):
    """Plot heatmap visualization using Seaborn library.

    Args:
        data_set (Pandas data frame): data set with features to be
    """
    utils.save_log('{0} :: {1}'.format(plot_heatmap.__module__,
                                       plot_heatmap.__name__))

    columns_to_print = deepcopy(features_engineering.features_list)
    columns_to_print.append(features_engineering.target_label)

    corr_matrix = data_set[columns_to_print].corr()
    k = 70  # number of variables for heat-map
    cols = corr_matrix.nlargest(
        k,
        features_engineering.target_label)[features_engineering.
                                           target_label]. \
        index
    cm = numpy.corrcoef(data_set[cols].values.T)
    sns.set(font_scale=1.25,
            rc={'figure.figsize': (15, 15)})
    sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.3f',
                annot_kws={'size': 8},
                yticklabels=cols.values,
                xticklabels=cols.values)
    plt.show()


def plot_feature_importance(cat_boost_model,
                            x_data_set,
                            y_data_set,
                            categorical_features_dims):
    """Plot feature importance learning with cat boost.
    Args:
        - x_data_set (pandas data frame):
        - y_data_set (pandas data frame):
        - categorical_features_dims (int):
    """
    utils.save_log('{0} :: {1}'.format(plot_feature_importance.__module__,
                                       plot_feature_importance.__name__))
    shap.initjs()
    shap_values = cat_boost_model.get_feature_importance(
        Pool(x_data_set,
             y_data_set,
             cat_features=categorical_features_dims),
        type='ShapValues')

    expected_value = shap_values[0, -1]
    shap_values = shap_values[:, :-1]
    # visualize the first prediction's explanation
    shap.force_plot(expected_value,
                    shap_values[200, :],
                    x_data_set.iloc[200, :])
    plt.show()
