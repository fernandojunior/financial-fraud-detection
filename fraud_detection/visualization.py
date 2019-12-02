import config as cfg
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import learning_curve
import numpy as np
import pandas as pd


def plot_heatmap(data):
    data_pd = data.toPandas()
    corr_matrix = data_pd.corr()
    k = 15  # number of variables for heatmap
    cols = corr_matrix.nlargest(k, cfg.LABEL)[cfg.LABEL].index
    cm = np.corrcoef(data_pd[cols].values.T)
    sns.set(font_scale=1.25, rc={'figure.figsize': (8, 8)})
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.3f', annot_kws={'size': 8}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def plot_learning_curve(estimator, x_data, y_data, title, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 10),
                        x_label="Training examples", y_label="Score"):
    '''
    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    '''
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, x_data, y_data, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_transactions_proportions(data, column="ProductCategory"):
    genuine_condition = "FraudResult == 0"
    fraud_condition = "FraudResult == 1"
    gen_column_name = "gen_count"
    fraud_column_name = "fraud_count"
    default_column_name = "count"
    x_var = "value"

    genuine_data = data.filter(genuine_condition).groupBy(column).count()
    genuine_data = genuine_data.withColumnRenamed(default_column_name, gen_column_name)

    fraudulent_data = data.filter(fraud_condition).groupBy(column).count()
    fraudulent_data = fraudulent_data.withColumnRenamed(default_column_name, fraud_column_name)

    aggregated_data_spark = genuine_data.join(fraudulent_data, on=[column], how='left_outer')
    aggregated_data_spark = aggregated_data_spark.fillna({fraud_column_name: '0'})

    aggregate_data_pd = aggregated_data_spark.toPandas()
    aggregate_data_pd[gen_column_name] = aggregate_data_pd[gen_column_name]/sum(aggregate_data_pd[gen_column_name])
    aggregate_data_pd[fraud_column_name] = aggregate_data_pd[fraud_column_name]/sum(aggregate_data_pd[fraud_column_name])
    aggregate_data_pd = pd.melt(aggregate_data_pd, id_vars=column,
                                value_vars=[fraud_column_name, gen_column_name], value_name=x_var)

    sns.catplot(y=column, hue='variable', x=x_var, kind='bar', data=aggregate_data_pd)
    plt.show()


def plot_roc(file_name):
    data = pd.read_csv(file_name, sep='\t')
    x = data['FPR']
    y = data['TPR']
    # This is the ROC curve
    plt.plot(x, y)
    plt.show()

    # This is the AUC
    auc = np.trapz(y, x)
    print('The AUC was: {0:.2f}.'.format(auc))


def plot_performance_comparison(data, x='max', y='score', hue='contamination'):
    data[x] = data[x].astype(np.int)
    data[y] = data[y].astype(np.float)
    data[hue] = data[hue].astype(np.float)
    sns.set(style='darkgrid')
    sns.lineplot(x=x, y=y, hue=hue, data=data)
    better_performance = data[data.score == data.score.max()].iloc[0]
    title = 'Better score: {0:.1f}\nMax sample: {1}\nContamination level: {2}'.format(
        better_performance[y], better_performance[x], better_performance[hue]
    )
    plt.title(title)
    plt.show()


def plot_hist(data, feature_columns, label_type):
    column = 'FraudResult == {0}'.format('1' if label_type else '0')
    data.filter(column).toPandas().hist(column=feature_columns, figsize=(15, 15))
    plt.show()


def plot_correlation(data, label='FraudResult', k=15):
    '''
    :param data:
    :param label:
    :param k: number of variables for heatmap.
    :return:
    '''
    data_pd = data.toPandas()
    corr_matrix = data_pd.corr()
    cols = corr_matrix.nlargest(k, label)[label].index
    cm = np.corrcoef(data_pd[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.1f', annot_kws={'size': 8}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()


def plot_bar(data, title):
    pd.Series(data).value_counts().plot.bar(title=title)
    plt.show()