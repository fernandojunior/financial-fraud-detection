from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.functions import (mean, dayofmonth, hour, dayofweek,
                                                                   month, weekofyear, dayofyear,
                                                                   format_number)

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


def get_transactions_list(data):
    return [item[1][0] for item in data.select('TransactionId').toPandas().iterrows()]