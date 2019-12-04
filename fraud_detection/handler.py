import config as cfg
import pandas as pd
import pyspark.sql.functions as F
from pyspark.sql.functions import (when, col, mean, 
                                   dayofmonth, hour, dayofweek,
                                   month, weekofyear, dayofyear)
import seaborn as sns

import logging
logging.basicConfig(filename='log_file.log', 
                    level=logging.INFO, 
                    format='%(asctime)s :: %(filename)s :: %(funcName)s ::\n\t%(message)s')
def outside_log(s1, s2):
    logging.info(s1 + '.py :: ' + s2)

#-----------------------------EXTRACT DATASET

def extract_data(**kwargs):
    logging.info('Extracting Data')
    cfg.data_train = read_data(kwargs['train_file_name'])

def read_data(file_name):
    """
    Return data in spark dataframe format.
    :param file_name: Input file name for dataset, could be a url or
    a path to a local file.
    :return: spark dataframe.
    """
    data = pd.read_csv(file_name)
    return cfg.spark.createDataFrame(data)

#-----------------------------CHECK MISSING DATA

def is_missing_data(data):
    ans = (cfg.data_train.count() != cfg.data_train.na.drop(how='any').count())
    return ans

#-----------------------------GET CONTAMINATION

def get_contamination(**kwargs):
    logging.info('Get contamination into the data')
    cfg.contamination_level = \
        (cfg.data_train.filter('FraudResult==1').count())/(cfg.data_train.count())

#-----------------------------CREATE NEW FEATURES

def create_features(**kwargs):
    logging.info('Creating pre-defined features')
    cfg.data_train = generate_new_features(cfg.data_train)

def generate_new_features(data):
    data = get_typeOfOperation(data)
    data = get_valueStrategy(data)
    data = get_timestamp(data)
    data = get_valuePerPeriod(data)
    data = get_featuresPerValue(data)
    return data

def get_typeOfOperation(data):
    data = data.withColumn("Operation", F.when(data.Amount > 0, 1)\
                                         .when(data.Amount < 0, -1).otherwise(0))
    return data

def get_valueStrategy(data):
    avg_value = data.agg({cfg.COLUMN_NAME: 'avg'}).collect()[0][0]
    data = data.withColumn('ValueStrategy',
          when(col(cfg.COLUMN_NAME) > avg_value * 1000, 3)
         .when(col(cfg.COLUMN_NAME) > avg_value * 100, 2)
         .when(col(cfg.COLUMN_NAME) > avg_value * 10, 1)
         .otherwise(0))
    return data

def get_timestamp(data):
    data = data.withColumn('Hour', 
                            hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', 
                            dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', 
                            dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', 
                            weekofyear(data['TransactionStartTime']))
    return data

def get_valuePerPeriod(data):
    data = data.withColumn('Vl_per_weekYr', 
                            (data['Value'] / data['WeekOfYear']))
    data = data.withColumn('Vl_per_dayWk',  
                            (data['Value'] / data['DayOfWeek']))
    data = data.withColumn('Vl_per_dayYr',  
                            (data['Value'] / data['DayOfYear']))
    return data

def get_featuresPerValue(data):
    for item in cfg.ITEMS_LIST:
        data = get_value_average(data, item)
        data = get_value_ratio(data, item)
    return data

def get_value_average(data, item):
    type_column = 'Avg'
    column_name = '{0}_vl_{1}'.format(type_column,item)
    aux = data.select([item,'Value']).groupBy(item).mean()
    aux = aux.select(col(item),col(type_column+'(Value)').alias(column_name))
    data = data.join(aux, on=item)
    return data

def get_value_ratio(data, item):
    column_name = 'Avg_vl_{0}'.format(item)
    ratio_column_name = 'Rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col('Value') - col(column_name)) / col(column_name))
    return data

#-----------------------------REMOVE FEATURES

def remove_features(**kwargs):
    logging.info('Removing features pre-defined')
    cfg.data_train = clean_data(cfg.data_train)

def clean_data(data, items_to_removed=cfg.ITEMS_TO_BE_REMOVED_LIST):
    return data.drop(*items_to_removed)

#-----------------------------SPLIT DATASET INTO TRAIN/VALID

def split_dataset_train(**kwargs):
    logging.info('Converting data to pandas and spliting it')
    data_pd = df_toPandas(**kwargs)
    separate_variables(data_pd, **kwargs)

def df_toPandas(**kwargs):
    logging.info(df_toPandas.__name__)
    return cfg.data_train.toPandas()

def separate_variables(data, **kwargs):
    cfg.x_train = data[cfg.ALL_FEATURES]
    cfg.y_train = data[cfg.LABEL]
    cfg.x_outliers = data[data[cfg.LABEL].isin([1])]
    cfg.x_train_numerical = cfg.x_train[cfg.NUMERICAL_FEATURES]
    cfg.x_outliers_numerical = cfg.x_outliers[cfg.NUMERICAL_FEATURES]

#-----------------------------CREATE FEATURE BY COUNTING OUTLIER FEATURES 

def createOutlierFeatures(**kwargs):
    logging.info('Creating outliers detections features')
    cfg.x_train[cfg.COUNT_COLUMN_NAME] = (cfg.x_train.IsolationForest + 
                                          cfg.x_train.LSCP + 
                                          cfg.x_train.KNN)
    add_OutlierFeatures_toList(**kwargs)

def add_OutlierFeatures_toList(**kwargs):
    new_features_list = [cfg.IF_COLUMN_NAME, 
                         cfg.LSCP_COLUMN_NAME, 
                         cfg.KNN_COLUMN_NAME, 
                         cfg.COUNT_COLUMN_NAME]
    cfg.CATEGORICAL_FEATURES += new_features_list
    cfg.ALL_FEATURES += new_features_list
    cfg.categorical_features_dims = [cfg.x_train.columns.get_loc(i) for i in cfg.CATEGORICAL_FEATURES[:]]
    cfg.numerical_features_dims = [cfg.x_train.columns.get_loc(i) for i in cfg.NUMERICAL_FEATURES[:]]

#-----------------------------BALANCE DATA

def balance_oversampling(**kwargs):
    logging.info('Balancing the train data')
    from models import smotenc_oversampling
    x, y = smotenc_oversampling()
    cfg.x_train_balanced = pd.DataFrame(x, columns=cfg.ALL_FEATURES)
    cfg.y_train_balanced = pd.DataFrame(y, columns=[cfg.LABEL])
    export_data_balanced(**kwargs)

def export_data_balanced(**kwargs):
    logging.info(export_data_balanced.__name__)
    cfg.x_train_balanced.to_csv(kwargs['output_x_file_name'], index=False)
    cfg.y_train_balanced.to_csv(kwargs['output_y_file_name'], index=False)

#-----------------------------

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
