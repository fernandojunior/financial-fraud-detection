import os.path
import pandas as pd
import sklearn.model_selection as sk_ml
from sklearn.metrics import precision_recall_fscore_support as score
from pyspark.sql.functions import (when, col, hour, dayofweek,
                                   weekofyear, dayofyear)

import config as cfg
import models
import visualization as vis
import utils as ut


LABEL = 'FraudResult'

ALL_FEATURES = ['ProviderId', 'ProductId', 'TransactionId',
                'BatchId', 'ProductCategory', 'ChannelId',
                'PricingStrategy', 'Value', 'Operation', 'Hour',
                'DayOfWeek', 'WeekOfYear',
                'Vl_per_weekYr', 'Vl_per_dayWk',
                'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']

NUMERICAL_FEATURES = ['Value', 'Operation', 'Hour',
                      'DayOfWeek', 'WeekOfYear',
                      'Vl_per_weekYr', 'Vl_per_dayWk',
                      'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']


def read_validation_data(**kwargs):
    """Read validation data.

    In this process, It is required the 'x' and 'y' data.
    They are saved in separately variables.

    Args:
        kwargs[validation_x_file_name] (str): validation vector.
        kwargs[validation_y_file_name] (str): target vector relative to X.
    """
    ut.save_log('{0} :: {1}'.format(read_validation_data.__module__,
                                    read_validation_data.__name__))

    cfg.x_validation = pd.read_csv(kwargs['output_valid_x_file'])
    cfg.y_validation = pd.read_csv(kwargs['output_valid_y_file'])
    cfg.x_to_predict_catboost = cfg.x_validation


def handle_data_test():
    ut.save_log('{0} :: {1}'.format(handle_data_test.__module__,
                                    handle_data_test.__name__))

    # create new features
    generate_new_features(cfg.data_test)
    cfg.ALL_FEATURES = cfg.ALL_FEATURES_TEST
    models.predict_isolation_forest()
    models.predict_lscp()
    models.predict_knn()
    create_feature_outlier_intensity()
    cfg.x_to_predict_catboost = cfg.x_data_temp


def generate_new_features(data):
    """This function is responsible for call multiples other functions
    to create the new features:

    Pipeline:
        - create_feature_is_credit_or_debit: create a new column identifying
        the transaction, 1 for debit and -1 for credit transactions
        - create_feature_how_huge_is_value: we use the mean of transactions
        to generate new features, identifying how distant the actual
        transaction is from the average.
        - create_features_based_on_timestamp_event:
        - get_value_per_period:
        - get_features_per_value:

    Args:
        data (spark data frame): this vector contains the data
        from Zindi challenge used to generate the new features.

    Returns:
        data (spark data frame): input data frame with data augmentation.
    """
    ut.save_log('{0} :: {1}'.format(generate_new_features.__module__,
                                    generate_new_features.__name__))

    data = create_feature_is_credit_or_debit(data)
    data = create_feature_how_huge_is_value(data)
    data = create_features_from_transaction_timestamp(data)
    data = create_feature_based_on_spent_by_timestamp(data)
    data = create_features_for_transaction_value_and_category(data)
    return data


def create_feature_is_credit_or_debit(data):
    """ This function create a new column (called Operation) that
    identify if the operation is credit (-1) or debit (1) transaction
    based on 'Amount' column.

    Args:
        data (spark data frame): input spark data frame.
    """
    ut.save_log('{0} :: {1}'.format(create_feature_is_credit_or_debit.__module__,
                                    create_feature_is_credit_or_debit.__name__))

    data = data.withColumn("Operation",
                           when(data.Amount > 0, 1).
                           when(data.Amount < 0, -1).
                           otherwise(0))
    return data


def create_feature_how_huge_is_value(data):
    """Create a feature identifying how distant is the transaction
    value from the average.

    Distances:
        - 3: the transaction is at least 100 times bigger than average.
        - 2: the transaction is at least 10 times bigger than average.
        - 1: the transaction is at least 2 times bigger than average.
        - 0: if none of the previous conditions are satisfied.

    Args:
        data (spark data frame): input spark data frame.
    """
    ut.save_log('{0} :: {1}'.format(create_feature_how_huge_is_value.__module__,
                                    create_feature_how_huge_is_value.__name__))

    avg_value = data.agg({cfg.COLUMN_VALUE: 'avg'}).collect()[0][0]
    data = data.withColumn('ValueStrategy',
                           when(col(cfg.COLUMN_VALUE) > avg_value * 100, 3).
                           when(col(cfg.COLUMN_VALUE) > avg_value * 10, 2).
                           when(col(cfg.COLUMN_VALUE) > avg_value * 2, 1).
                           otherwise(0))
    return data


def create_features_from_transaction_timestamp(data):
    """Each transaction has a specific timestamp identifying
    when this happened. So, new features are created based on
    this.

    Features:
        - Hour: hour time identifying the moment the
        transaction happened.
        - DayOfWeek: identify which day of week the
        transaction was made.
        - DayOfYear: identify which day of year the
        transaction was made.
        - WeekOfYear: identify which week of year the
        transaction was made.

    Args:
        data (spark data frame): input spark data frame.
    """
    ut.save_log('{0} :: {1}'.format(create_features_from_transaction_timestamp.__module__,
                                    create_features_from_transaction_timestamp.__name__))

    data = data.withColumn('Hour',
                           hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek',
                           dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear',
                           dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear',
                           weekofyear(data['TransactionStartTime']))
    return data


def create_feature_based_on_spent_by_timestamp(data):
    """Ratio between value information and some timestamp
    feature created previously.

    Features:
        - Vl_per_weekYr: ration between the transaction value
        and the week of year.
        - Vl_per_dayWk: ration between the transaction value
        and the day of week.
        - Vl_per_dayYr: ration between the transaction value
        and the day of year.

    Args:
        data (spark data frame): input spark data frame.
    """
    ut.save_log('{0} :: {1}'.format(create_feature_based_on_spent_by_timestamp.__module__,
                                    create_feature_based_on_spent_by_timestamp.__name__))

    data = data.withColumn('Vl_per_weekYr',
                           (data['Value'] / data['WeekOfYear']))
    data = data.withColumn('Vl_per_dayWk',
                           (data['Value'] / data['DayOfWeek']))
    data = data.withColumn('Vl_per_dayYr',
                           (data['Value'] / data['DayOfYear']))
    return data


def create_features_for_transaction_value_and_category(data):
    """Create new features relating the transaction value for
    each product category.

    Args:
        data (spark data frame): input spark data frame.
    """
    ut.save_log('{0} :: {1}'.format(create_features_for_transaction_value_and_category.__module__,
                                    create_features_for_transaction_value_and_category.__name__))

    list_of_categories = ['ProductId', 'ProviderId']
    for item in list_of_categories:
        data = create_feature_average_value_for_category(data, item)
        data = create_feature_ratio_between_value_and_category(data, item)
    return data


def create_feature_average_value_for_category(data, item):
    """Create a new feature with average transaction value for
    all aggregated by item.

    Args:
        data (spark data frame): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.

    Returns:
        data (spark data frame): output spark data frame with
        the new feature created.
    """
    ut.save_log('{0} :: {1}'.format(create_feature_average_value_for_category.__module__,
                                    create_feature_average_value_for_category.__name__))

    column_name = '{0}_vl_{1}'.format("avg", item)
    aux = data.select([item, 'Value']).\
        groupBy(item).\
        mean()
    aux = aux.select(col(item),
                     col('avg' +
                         '(Value)').alias(column_name))
    data = data.join(aux, on=item)
    return data


def create_feature_ratio_between_value_and_category(data, item):
    """Create a new feature with the ratio between the transaction
    value and the product category.

    Args:
        data (spark data frame): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.
    """
    ut.save_log('{0} :: {1}'.format(create_feature_ratio_between_value_and_category.__module__,
                                    create_feature_ratio_between_value_and_category.__name__))

    column_name = 'avg_vl_{0}'.format(item)
    ratio_column_name = 'rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col('Value') - col(column_name)) /
                           col(column_name))
    return data


def create_feature_outlier_intensity():
    """Each model (Isolation Forest, LSCP, and KNN) classify the instance
    into outlier or not. This function create a new feature with the number
    of models that classify the instance as outlier. For example:
    - if only KNN model classify the instance as outlier, this value will be 1
    - if LSCP and KNN models classify the instance as outlier, so this value
    will be 2
    - if all models classify the instance as outlier, so this value will be 3
    - if none model is outlier, so this value will be 0
    """
    ut.save_log('{0} :: {1}'.format(create_feature_outlier_intensity.__module__,
                                    create_feature_outlier_intensity.__name__))

    cfg.x_data_temp[cfg.COUNT_COLUMN_NAME] = \
        (cfg.x_data_temp.IsolationForest +
         cfg.x_data_temp.LSCP +
         cfg.x_data_temp.KNN)
    add_outlier_to_features_list()


def add_outlier_to_features_list():
    """The variables CATEGORICAL_FEATURES and ALL_FEATURES
    keep the relation of categorical features, so this
    function update these lists.
    """
    ut.save_log('{0} :: {1}'.format(add_outlier_to_features_list.__module__,
                                    add_outlier_to_features_list.__name__))

    new_features_list = [cfg.IF_COLUMN_NAME,
                         cfg.LSCP_COLUMN_NAME,
                         cfg.KNN_COLUMN_NAME,
                         cfg.COUNT_COLUMN_NAME]

    cfg.CATEGORICAL_FEATURES += new_features_list
    cfg.ALL_FEATURES += new_features_list
    cfg.categorical_features_dims = \
        [cfg.x_data_temp.columns.get_loc(i)
         for i in cfg.CATEGORICAL_FEATURES[:]]


def save_predictions_xente(file_name, transactions_list, test_pred):
    """
    :param file_name:
    :param transactions_list:
    :param test_pred:
    :return:
    """
    ut.save_log('{0} :: {1}'.format(save_predictions_xente.__module__,
                                    save_predictions_xente.__name__))

    file = open(file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for trans_id, value in zip(transactions_list, test_pred):
        file.write('{0},{1}\n'.format(trans_id, int(value)))
    file.close()