import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def plot_transactions_proportions(data):
    df1 = data.filter('FraudResult == 0').groupBy('ProductCategory' ).count()
    df1 = df1.withColumnRenamed('count', 'gen_count')

    df2 = data.filter('FraudResult == 1').groupBy('ProductCategory' ).count()
    df2 = df2.withColumnRenamed('count', 'fraud_count')

    new_df = df1.join(df2, on=['ProductCategory'], how='left_outer')
    new_df = new_df.fillna({'fraud_count': '0'})

    new_df_pd = new_df.toPandas()
    new_df_pd['gen_count'] = new_df_pd['gen_count']/sum(new_df_pd['gen_count'])
    new_df_pd['fraud_count'] = new_df_pd['fraud_count']/sum(new_df_pd['fraud_count'])
    new_df_pd = pd.melt(new_df_pd, id_vars='ProductCategory', value_vars=['fraud_count', 'gen_count'], value_name='value')

    sns.catplot(y='ProductCategory', hue='variable', x='value', kind='bar', data=new_df_pd)
    plt.show()