import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from catboost import Pool
import shap
from sklearn.model_selection import learning_curve
import utils as ut
import features_engineering as fte


def plot_target_distribution(y_data_set):
    """Plot visualization
    Args:
        - y_data_set (Pandas Data Frame)
    """
    ut.save_log('{0} :: {1}'.format(
        plot_target_distribution.__module__,
        plot_target_distribution.__name__))
    sns.set(font_scale=1.25, rc={'figure.figsize': (4, 4)})
    fg = pd.Series(y_data_set).\
        value_counts().plot.bar(title='SMOTENC Output')
    fg.plot()
    plt.show()


def plot_heatmap(data_set):
    """Plot heatmap visualization using Seaborn library.

    Args:
        data_set (Pandas data frame): data set with features to be
    """
    ut.save_log('{0} :: {1}'.format(plot_heatmap.__module__,
                                    plot_heatmap.__name__))

    corr_matrix = data_set.corr()
    k = 70  # number of variables for heat-map
    cols = corr_matrix.nlargest(k, fte.target_label)[fte.target_label].index
    cm = np.corrcoef(data_set[cols].values.T)
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
    ut.save_log('{0} :: {1}'.format(plot_feature_importance.__module__,
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


def plot_learning_curve(estimator, x_data, y_data, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10),
                        x_label="Training examples", y_label="Score"):
    ut.save_log('{0} :: {1}'.format(plot_learning_curve.__module__,
                                    plot_learning_curve.__name__))

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    train_sizes, train_scores, test_scores = \
        learning_curve(estimator,
                       x_data,
                       y_data,
                       cv=cv,
                       n_jobs=n_jobs,
                       train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes,
                     train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes,
                     test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1,
                     color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_transactions_proportions(data, column="ProductCategory"):
    """Plot transaction proportion
    Args:
        - data (pandas data frame):
    """
    ut.save_log('{0} :: {1}'.format(
        plot_transactions_proportions.__module__,
        plot_transactions_proportions.__name__))
    genuine_condition = "FraudResult == 0"
    fraud_condition = "FraudResult == 1"
    gen_column_name = "gen_count"
    fraud_column_name = "fraud_count"
    default_column_name = "count"
    x_var = "value"

    genuine_data = data.filter(genuine_condition).groupBy(column).count()
    genuine_data = genuine_data.withColumnRenamed(default_column_name,
                                                  gen_column_name)

    fraudulent_data = data.filter(fraud_condition).groupBy(column).count()
    fraudulent_data = fraudulent_data.withColumnRenamed(default_column_name,
                                                        fraud_column_name)

    aggregated_data_spark = genuine_data.join(fraudulent_data,
                                              on=[column],
                                              how='left_outer')
    aggregated_data_spark = \
        aggregated_data_spark.fillna({fraud_column_name: '0'})

    aggregate_data_pd = aggregated_data_spark.toPandas()
    aggregate_data_pd[gen_column_name] = \
        aggregate_data_pd[gen_column_name] /\
        sum(aggregate_data_pd[gen_column_name])
    aggregate_data_pd[fraud_column_name] = \
        aggregate_data_pd[fraud_column_name] /\
        sum(aggregate_data_pd[fraud_column_name])
    aggregate_data_pd = pd.melt(aggregate_data_pd,
                                id_vars=column,
                                value_vars=[fraud_column_name,
                                            gen_column_name],
                                value_name=x_var)

    sns.catplot(y=column,
                hue='variable',
                x=x_var,
                kind='bar',
                data=aggregate_data_pd)
    plt.show()


def plot_performance_comparison(data, x='max', y='score', hue='contamination'):
    """Plot performance for multiples params
    Args:
        - data (pandas dataframe):
    """
    ut.save_log('{0} :: {1}'.format(plot_performance_comparison.__module__,
                                    plot_performance_comparison.__name__))
    data[x] = data[x].astype(np.int)
    data[y] = data[y].astype(np.float)
    data[hue] = data[hue].astype(np.float)
    sns.set(style='darkgrid')
    sns.lineplot(x=x, y=y, hue=hue, data=data)
    better_performance = data[data.score == data.score.max()].iloc[0]
    title = 'Better score: {0:.1f}\n' \
            'Max sample: {1}\n' \
            'Contamination level: {2}'.format(better_performance[y],
                                              better_performance[x],
                                              better_performance[hue])
    plt.title(title)
    plt.show()


def plot_hist(data, feature_columns, label_type):
    """Plot histogram for label classes.
    Args:
        - data (pandas dataframe)
    """
    ut.save_log('{0} :: {1}'.format(plot_hist.__module__,
                                    plot_hist.__name__))

    column = 'FraudResult == {0}'.format('1' if label_type else '0')
    data.filter(column).toPandas().hist(column=feature_columns,
                                        figsize=(15, 15))
    plt.show()


def plot_correlation(data, label='FraudResult', k=15):
    """
    :param data:
    :param label:
    :param k: number of variables for heatmap.
    :return:
    """
    ut.save_log('{0} :: {1}'.format(plot_correlation.__module__,
                                    plot_correlation.__name__))
    data_pd = data.toPandas()
    corr_matrix = data_pd.corr()
    cols = corr_matrix.nlargest(k, label)[label].index
    cm = np.corrcoef(data_pd[cols].values.T)
    sns.set(font_scale=1.25)
    sns.heatmap(cm,
                cbar=True,
                annot=True,
                square=True,
                fmt='.1f',
                annot_kws={'size': 8},
                yticklabels=cols.values,
                xticklabels=cols.values)
    plt.show()


def plot_bar(data, title):
    """Plot bar to show the (im)balanced distribution.
    Args:
        - data (pandas data frame)
    """
    ut.save_log('{0} :: {1}'.format(plot_bar.__module__,
                                    plot_bar.__name__))
    pd.Series(data).value_counts().plot.bar(title=title)
    plt.show()
