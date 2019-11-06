import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_transactions_proportions(data):
    genuine_data = data.filter('FraudResult == 0').groupBy('ProductCategory' ).count()
    genuine_data = genuine_data.withColumnRenamed('count', 'gen_count')

    fraudulent_data = data.filter('FraudResult == 1').groupBy('ProductCategory' ).count()
    fraudulent_data = fraudulent_data.withColumnRenamed('count', 'fraud_count')

    aggregated_data_spark = genuine_data.join(fraudulent_data, on=['ProductCategory'], how='left_outer')
    aggregated_data_spark = aggregated_data_spark.fillna({'fraud_count': '0'})

    aggregate_data_pd = aggregated_data_spark.toPandas()
    aggregate_data_pd['gen_count'] = aggregate_data_pd['gen_count']/sum(aggregate_data_pd['gen_count'])
    aggregate_data_pd['fraud_count'] = aggregate_data_pd['fraud_count']/sum(aggregate_data_pd['fraud_count'])
    aggregate_data_pd = pd.melt(aggregate_data_pd, id_vars='ProductCategory',
                                value_vars=['fraud_count', 'gen_count'], value_name='value')

    sns.catplot(y='ProductCategory', hue='variable', x='value', kind='bar', data=aggregate_data_pd)
    plt.show()


def plot_performance_comparison(data, x='max', y='score', hue='contamination'):
    sns.set(style='darkgrid')
    sns.lineplot(x=x, y=y, hue=hue, data=data)
    better_performance = data[data.score == data.score.max()].iloc[0]
    print(data)
    print(better_performance)
    title = 'Better score: {0:.1f}\nMax sample: {1}\nContamination level: {2}'.format(
        better_performance[y], better_performance[x], better_performance[hue]
    )
    plt.title(title)
    plt.show()


def plot_hist(data, feature_columns, label_type):
    column = 'FraudResult == {0}'.format('1' if label_type else '0')
    data.filter(column).toPandas().hist(column=feature_columns, figsize=(5, 5))
    plt.show()


def plot_heatmap(data, numerical_features, label_type, method='spearman'):
    column = 'FraudResult == {0}'.format('1' if label_type else '0')
    data = data.filter(column).select(numerical_features).toPandas()
    corr = data.corr(method)
    ax = sns.heatmap(
        corr, vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True)
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45, horizontalalignment='right')
    plt.show()


def plot_bar(data, title):
    pd.Series(data).value_counts().plot.bar(title=title)
    plt.show()