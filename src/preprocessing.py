from pyspark.sql.functions import col
import pyspark.sql.functions as F


def there_is_missing_data(data):
    ans = data.count() != data.na.drop(how='any').count()
    print('There is missing data? {0}'.format(ans))


def there_is_duplicate_lines(data):
    ans = data.count() != data.distinct().count()
    print('There is duplicated lines? {0}'.format(ans))


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

    gen_train_data = data.filter('FraudResult == 0')
    items_list = ['ChannelId', 'ProductCategory', 'ProductId']

    for item in items_list:
        mean_column_name = 'avg_ps_{0}'.format(item)
        ratio_column_name = 'rt_avg_ps_{0}'.format(item)
        aux = gen_train_data.select([item, 'PositiveAmount']).groupBy(item).mean()
        aux = aux.select(col(item), col('avg(PositiveAmount)').alias(mean_column_name))
        data = data.join(aux, on=item)
        data = data.withColumn(ratio_column_name,
                               (F.col('PositiveAmount') - F.col(mean_column_name)) / F.col(
                                               mean_column_name))
    return data
