import config as cfg
import pandas as pd
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import (mean, dayofmonth, hour, dayofweek,
                                                                   month, weekofyear, dayofyear,
                                                                   format_number)
from imblearn.over_sampling import SMOTENC
import seaborn as sns

def read_data(file_name):
    """
    Return data in spark dataframe format.
    :param file_name: Input file name for dataset, could be a url or
    a path to a local file.
    :return: spark dataframe.
    """
    data = pd.read_csv(file_name)
    spark_data = cfg.spark.createDataFrame(data)
    return spark_data


def there_is_missing_data(data):
    ans = data.count() != data.na.drop(how='any').count()
    print('There is missing data? {0}'.format(ans))


def there_is_duplicate_lines(data):
    ans = data.count() != data.distinct().count()
    print('There is duplicated lines? {0}'.format(ans))


def get_specific_statistical_info(data, stat):
    """
    :param data: PySpark data frame.
    :param stat: could be "avg", "max", or "min".
    :return: The respective statistics over input data.
    """
    if stat == "avg":
        data = data.mean()
    elif stat == "max":
        data = data.max()
    elif stat == "min":
        data = data.min()
    return data


def get_features_augmentation(data):
    '''
    This function was create to make features augmentation in the data to improve
    the performance model.
    :param data:
    :return:
        - Operation: The Amount value corresponds to transaction value,
        if positive value indicates a debit transaction, while a negative
        value indicates a credit transaction. So, we map this OPERATION
        to -1 and 1, in which -1 was used for credit and 1 for debit.
        - PositiveAmount: There is some difference between the Amount and
        Value column, Value is suppose to be the absolute value of Amount.
        So, we calculate the PositiveAmount, which corresponds to the
        absolute value of Amount, after that we discard this attribute.
        - We had three categorical features: ChannelId, ProductCategory, and
        ProductId. We decide create new features based on this, they are
        repeated process for the three categories:
            - Avg_ps: average of positive amount for a same category product,
            e.g., there is different channels, each channels has a mean value
            for all transactions of this category, so a new column was added
            to data frame including this value, indicating the mean value for
            this category product.
            - Rt_avg_ps: this feature calculate the ratio between the transaction
             value instance and the global mean. How
    '''
    data = data.withColumn("Operation",
                           F.when(data.Amount > 0, 1).when(data.Amount < 0, -1).otherwise(0))
    data = data.withColumn('PositiveAmount', F.abs(data['Amount']))

    data = data.withColumn('Hour', hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', weekofyear(data['TransactionStartTime']))
    data = data.withColumn('Month', month(data['TransactionStartTime']))

    data = data.withColumn('Ps_per_dayWk', (data['PositiveAmount'] / data['DayOfWeek']))
    data = data.withColumn('Ps_per_dayYr', (data['PositiveAmount'] / data['DayOfYear']))
    data = data.withColumn('Op_x_value', (data['Operation'] * data['Value']))

    items_list = ['ChannelId', 'ProductCategory', 'ProductId']
    for item in items_list:
        for statistical_type in ["avg", "min", "max"]:
            column_name = "{0}_ps_{1}".format(statistical_type, item)
            aux = get_specific_statistical_info(
                data.select([item, 'PositiveAmount']).groupBy(item), statistical_type)
            aux = aux.select(col(item), col("{0}(PositiveAmount)".format(statistical_type)).alias(column_name))
            data = data.join(aux, on=item)
            if statistical_type == "avg":
                ratio_column_name = 'rt_avg_ps_{0}'.format(item)
                data = data.withColumn(ratio_column_name,
                                       (F.col('PositiveAmount') - F.col(column_name)) / F.col(column_name))
    return data


def get_amount_based_features(data):
    avg_value = data.agg({cfg.COLUMN_NAME: 'avg'}).collect()[0][0]
    data = data.withColumn(
        'ValueStrategy',
        F.when(F.col(cfg.COLUMN_NAME) > avg_value * 1000, 3)
        .when(F.col(cfg.COLUMN_NAME) > avg_value * 100, 2)
        .when(F.col(cfg.COLUMN_NAME) > avg_value * 10, 1)
        .otherwise(0)
    )
    return data


def get_operation_based_features(data):
    data = data.withColumn("Operation",
                           F.when(data.Amount > 0, 1)
                           .when(data.Amount < 0, -1).otherwise(0))
    return data


def get_time_based_features(data):
    data = data.withColumn('Hour', hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', F.dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', weekofyear(data['TransactionStartTime']))

    data = data.withColumn('Vl_per_weekYr', (data['Value'] / data['WeekOfYear']))
    data = data.withColumn('Vl_per_dayWk', (data['Value'] / data['DayOfWeek']))
    data = data.withColumn('Vl_per_dayYr', (data['Value'] / data['DayOfYear']))
    return data


def get_average_value_based_features(data, item):
    mean_column_name = 'avg_vl_{0}'.format(item)
    mean_aux = data.select([item, 'Value']).groupBy(item).mean()
    mean_aux = mean_aux.select(col(item), col('avg(Value)').alias(mean_column_name))
    data = data.join(mean_aux, on=item)
    return data


def get_value_ratio_based_features(data, item):
    mean_column_name = 'avg_vl_{0}'.format(item)
    ratio_column_name = 'Rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (F.col('Value') - F.col(mean_column_name)) / F.col(mean_column_name))
    return data


def create_variations_based_on_value(data):
    for item in cfg.ITEMS_LIST:
        data = get_average_value_based_features(data, item)
        data = get_value_ratio_based_features(data, item)
    return data


def generate_new_features(data):
    """
    :param data:
    :return:
    """
    cfg.contamination_level = (data.filter('FraudResult==1').count()) / (data.count())
    data = get_amount_based_features(data)
    data = get_operation_based_features(data)
    data = get_time_based_features(data)
    data = create_variations_based_on_value(data)
    return data


def clean_data(data, items_to_removed=cfg.ITEMS_TO_BE_REMOVED_LIST):
    return data.drop(*items_to_removed)


def get_transactions_list(data):
    return [item[1][0] for item in data.select('TransactionId').toPandas().iterrows()]


def add_features(data):
    data[cfg.COUNT_COLUMN_NAME] = (data.IsolationForest + data.LSCP + data.KNN)
    new_features_list = [cfg.IF_COLUMN_NAME, cfg.LSCP_COLUMN_NAME, cfg.KNN_COLUMN_NAME, cfg.COUNT_COLUMN_NAME]
    cfg.categorical_features += new_features_list
    cfg.ALL_FEATURES += new_features_list
    cfg.categorical_features_dims = [data.columns.get_loc(i) for i in cfg.CATEGORICAL_FEATURES[:]]
    cfg.numerical_features_dims = [data.columns.get_loc(i) for i in cfg.NUMERICAL_FEATURES[:]]


def separate_variables(data):
    data_df = data.toPandas()
    cfg.x_train = data_df[cfg.ALL_FEATURES]
    cfg.y_train = data_df[cfg.LABEL]
    cfg.x_outliers = data_df[data_df[cfg.LABEL].isin([1])]
    cfg.x_train_numerical = cfg.x_train[cfg.NUMERICAL_FEATURES]
    cfg.x_outliers_numerical = cfg.x_outliers[cfg.NUMERICAL_FEATURES]


def balance_data(data):
    data = data[cfg.ALL_FEATURES]
    sm = SMOTENC(categorical_features=cfg.categorical_features_dims, random_state=42, n_jobs=10)
    x_smotenc, y_smotenc = sm.fit_sample(data, cfg.y_train)
    x_smotenc = pd.DataFrame(x_smotenc, columns=cfg.ALL_FEATURES)
    y_smotenc = pd.DataFrame(y_smotenc, columns=[cfg.LABEL])
    sns.set(font_scale=1.25, rc={'figure.figsize': (4, 4)})
    pd.Series(y_smotenc[cfg.LABEL]).value_counts().plot.bar(title='SMOTENC : Count - Fraud Result')


def save_predictions_xente(file_name, transactions_list, test_pred):
    """
    :param file_name:
    :param transactions_list:
    :param test_pred:
    :return:
    """
    file = open(file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for trans_id, value in zip(transactions_list, test_pred):
        file.write('{0},{1}\n'.format(trans_id, int(value)))
    file.close()
