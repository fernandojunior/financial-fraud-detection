from pyspark.sql.functions import col
import pyspark.sql.functions as F


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


def get_features_augmentation(data, gen_train_data=[]):
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
    data = data.withColumn("Operation", F.when(data.Amount > 0, 1).when(data.Amount < 0, -1).otherwise(0))
    data = data.withColumn('PositiveAmount', F.abs(data['Amount']))
    
    data = data.withColumn('Hour', F.hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', F.dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', F.dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', F.weekofyear(data['TransactionStartTime']))
    data = data.withColumn('Month', F.month(data['TransactionStartTime']))    

    data = data.withColumn('Ps_pr_dayWeek_pr_Month', ( data['PositiveAmount']/data['DayOfWeek']/data['Month'] ) )
    data = data.withColumn('Ps_pr_dayYear_pr_WeekYear', ( data['PositiveAmount']/data['DayOfYear']/data['WeekOfYear'] ) )
    data = data.withColumn('OpCredSum_pr_Month', ( data.filter('Operation==1').count()/data['Month'] ) )
    data = data.withColumn('OpDebtSum_pr_Month', ( data.filter('Operation==-1').count()/data['Month'] ) )

    gen_train_data = gen_train_data if gen_train_data else data.filter('FraudResult == 0')
    items_list = ['AccountId','ChannelId','ProductCategory','ProductId','Hour','DayOfYear','WeekOfYear','Month']
    for item in items_list:
        mean_column_name = 'avg_ps_{0}'.format(item)
        mean_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).mean()
        mean_aux = mean_aux.select(col(item), F.col('avg(PositiveAmount)').alias(mean_column_name))
        data = data.join(mean_aux, on=item)

        min_column_name = 'min_ps_{0}'.format(item)
        min_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).min()    
        min_aux = min_aux.select(col(item), F.col('min(PositiveAmount)').alias(min_column_name))
        data= data.join(min_aux, on=item)

        max_column_name = 'max_ps_{0}'.format(item)
        max_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).max()    
        max_aux = max_aux.select(col(item), F.col('max(PositiveAmount)').alias(max_column_name))
        data = data.join(max_aux, on=item)
        
        mean_pi_column_name = 'pi_{0}'.format(item)
        mean_pi = gen_train_data.select([item,'PricingStrategy']).groupBy(item).mean()
        mean_pi = mean_pi.select(col(item), F.col('avg(PricingStrategy)').alias(mean_pi_column_name))
        data = data.join(mean_pi, on=item)
        
        ratio_column_name = 'rt_avg_ps_{0}'.format(item)
        data = data.withColumn(ratio_column_name, (F.col('PositiveAmount')-F.col(mean_column_name))/ F.col(mean_column_name))
        
    return data
    
def get_features_augmentation_test(data, gen_train_data=[]):
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
    data = data.withColumn("Operation", F.when(data.Amount > 0, 1).when(data.Amount < 0, -1).otherwise(0))
    data = data.withColumn('PositiveAmount', F.abs(data['Amount']))
    
    data = data.withColumn('Hour', F.hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', F.dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', F.dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', F.weekofyear(data['TransactionStartTime']))
    data = data.withColumn('Month', F.month(data['TransactionStartTime']))    

    data = data.withColumn('Ps_pr_dayWeek_pr_Month', ( data['PositiveAmount']/data['DayOfWeek']/data['Month'] ) )
    data = data.withColumn('Ps_pr_dayYear_pr_WeekYear', ( data['PositiveAmount']/data['DayOfYear']/data['WeekOfYear'] ) )
    data = data.withColumn('OpCredSum_pr_Month', ( data.filter('Operation==1').count()/data['Month'] ) )
    data = data.withColumn('OpDebtSum_pr_Month', ( data.filter('Operation==-1').count()/data['Month'] ) )
    
    gen_train_data = gen_train_data if gen_train_data else data
    items_list = ['AccountId','ChannelId','ProductCategory','ProductId','Hour','DayOfYear','WeekOfYear','Month']
    for item in items_list:
        mean_column_name = 'avg_ps_{0}'.format(item)
        mean_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).mean()
        mean_aux = mean_aux.select(col(item), F.col('avg(PositiveAmount)').alias(mean_column_name))
        data = data.join(mean_aux, on=item)

        min_column_name = 'min_ps_{0}'.format(item)
        min_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).min()    
        min_aux = min_aux.select(col(item), F.col('min(PositiveAmount)').alias(min_column_name))
        data= data.join(min_aux, on=item)

        max_column_name = 'max_ps_{0}'.format(item)
        max_aux = gen_train_data.select([item,'PositiveAmount']).groupBy(item).max()    
        max_aux = max_aux.select(col(item), F.col('max(PositiveAmount)').alias(max_column_name))
        data = data.join(max_aux, on=item)
        
        mean_pi_column_name = 'pi_{0}'.format(item)
        mean_pi = gen_train_data.select([item,'PricingStrategy']).groupBy(item).mean()
        mean_pi = mean_pi.select(col(item), F.col('avg(PricingStrategy)').alias(mean_pi_column_name))
        data = data.join(mean_pi, on=item)
        
        ratio_column_name = 'rt_avg_ps_{0}'.format(item)
        data = data.withColumn(ratio_column_name, (F.col('PositiveAmount')-F.col(mean_column_name))/ F.col(mean_column_name))
        
    return data


def get_transactions_list(data):
    return [item[1][0] for item in data.select('TransactionId').toPandas().iterrows()]
