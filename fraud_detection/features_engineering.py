from pyspark.sql.functions import (when, col, hour, dayofweek,
                                   weekofyear, dayofyear)

import utils
import config

target_label = config.target_column_name[0]

features_list = config.feature_categorical_to_check_spent_value +\
    config.others_categorical_columns_list +\
    config.others_numerical_columns_list +\
    config.feature_column_value

categorical_features_list = config.feature_categorical_to_check_spent_value +\
    config.others_categorical_columns_list

categorical_features_dims = 0

numerical_features_list = config.feature_column_value +\
    config.others_numerical_columns_list

fraudulent_percentage = 0


def generate_new_features(data):
    """Caller multiples functions
        to create the new features on spark dataframe.

    Args:
        data (spark dataframe): a matrix dataframe

    Returns:
        data (spark dataframe): a matrix dataframe with new features.
    """
    utils.save_log('{0} :: {1}'.format(
        generate_new_features.__module__,
        generate_new_features.__name__))

    data = create_feature_is_credit_debit(data)
    data = create_feature_value_category(data)
    data = create_features_from_transaction_timestamp(data)
    data = create_feature_based_on_spent_by_timestamp(data)
    list_of_categories = config.feature_categorical_to_check_spent_value
    data = create_features_avg_ratio_value_by_categories(data,
                                                         list_of_categories)
    return data


def create_feature_is_credit_debit(data):
    """ Create new column Operation based on Amount column
        Identifier if the operation is:
        - credit (-1) or debit (1)

    Args:
        data (spark data frame): input spark data frame.
    """
    utils.save_log('{0} :: {1}'.format(
        create_feature_is_credit_debit.__module__,
        create_feature_is_credit_debit.__name__))

    data = data.withColumn("Operation",
                           when(data.Amount > 0, 1).
                           otherwise(0))

    update_list_features("numerical", ["Operation"])

    return data


def create_feature_value_category(data):
    """Categorizer of transaction value based on
        distance to the average transaction value of whole dataframe.

    Distances:
        - 3: at least 100 times bigger than average.
        - 2: at least 10 times bigger than average.
        - 1: at least 2 times bigger than average.
        - 0: if none of the previous conditions are satisfied.

    Args:
        data (spark dataframe): input spark data frame.
    """
    utils.save_log('{0} :: {1}'.format(
        create_feature_value_category.__module__,
        create_feature_value_category.__name__))

    avg_value = data.agg({'Value': 'avg'}).collect()[0][0]
    data = data. \
        withColumn('ValueStrategy',
                   when(col('Value') > avg_value * 200, 2).
                   when(col('Value') > avg_value * 50, 1).
                   otherwise(0))

    update_list_features("numerical", ["ValueStrategy"])

    return data


def create_features_from_transaction_timestamp(data):
    """Extraction of transact timestamp.
        New features are created based on.

    Features:
        - Hour: hour time transaction
        - DayOfWeek: day of week transaction
        - DayOfYear: day of year transaction
        - WeekOfYear: week of year transaction

    Args:
        data (spark dataframe): input spark data frame.
    """
    utils.save_log('{0} :: {1}'.format(
        create_features_from_transaction_timestamp.__module__,
        create_features_from_transaction_timestamp.__name__))

    data = data.withColumn('TransactionHour',
                           hour(data[config.feature_column_timestamp]))
    data = data.withColumn('TransactionDayOfWeek',
                           dayofweek(data[config.feature_column_timestamp]))
    data = data.withColumn('TransactionDayOfYear',
                           dayofyear(data[config.feature_column_timestamp]))
    data = data.withColumn('TransactionWeekOfYear',
                           weekofyear(data[config.feature_column_timestamp]))

    data = data.withColumn('WeekAction',
                           when(col('TransactionWeekOfYear').
                                between(50, 52), 1).
                           otherwise(0))

    update_list_features("numerical", ['TransactionHour',
                                       'TransactionDayOfWeek',
                                       'TransactionDayOfYear',
                                       'TransactionWeekOfYear',
                                       'WeekAction'])

    return data


def create_feature_based_on_spent_by_timestamp(data):
    """Ratio value by timestamp transaction

    Args:
        data (spark dataframe): input spark data frame.
    """
    utils.save_log('{0} :: {1}'.format(
        create_feature_based_on_spent_by_timestamp.__module__,
        create_feature_based_on_spent_by_timestamp.__name__))

    data = data.withColumn('RatioValueSpentByWeekOfYear',
                           (data['Value'] / data['TransactionWeekOfYear']))
    data = data.withColumn('RatioValueSpentByDayOfWeek',
                           (data['Value'] / data['TransactionDayOfWeek']))
    data = data.withColumn('RatioValueSpentByDayOfYear',
                           (data['Value'] / data['TransactionDayOfYear']))

    update_list_features("numerical", ['RatioValueSpentByWeekOfYear',
                                       'RatioValueSpentByDayOfWeek',
                                       'RatioValueSpentByDayOfYear'])

    return data


def create_features_avg_ratio_value_by_categories(data, list_of_categories):
    """Create new features relating the transaction value for
    each product category.

    Args:
        data: input spark data frame.
        list_of_categories: features to be inserted on global features list
    """
    utils.save_log('{0} :: {1}'.format(
        create_features_avg_ratio_value_by_categories.__module__,
        create_features_avg_ratio_value_by_categories.__name__))

    for item in list_of_categories:
        data = create_feature_average_value_for_category(data, item)
        data = create_feature_ratio_between_value_and_category(data, item)
    return data


def create_feature_average_value_for_category(data, item):
    """Create feature based on average value transaction per
        all aggregated by item.

    Args:
        data (spark dataframe): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.

    Returns:
        data (spark data frame): output spark data frame with
        the new feature created.
    """
    utils.save_log('{0} :: {1}'.format(
        create_feature_average_value_for_category.__module__,
        create_feature_average_value_for_category.__name__))

    column_name = 'AverageValuePer{0}'.format(item)
    aux = data.select([item, config.feature_column_value[0]]).\
        groupBy(item).\
        mean()
    aux = aux.select(col(item),
                     col('avg' +
                         '(Value)').alias(column_name))
    data = data.join(aux, on=item)
    update_list_features("numerical", [column_name])
    return data


def create_feature_ratio_between_value_and_category(data, item):
    """Create a new feature with the ratio between the transaction
        value and the feature category.

    Args:
        data (spark data frame): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.
    """
    utils.save_log('{0} :: {1}'.format(
        create_feature_ratio_between_value_and_category.__module__,
        create_feature_ratio_between_value_and_category.__name__))

    column_name = 'AverageValuePer{0}'.format(item)
    ratio_column_name = 'RatioOfAverageValuePer{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col(config.feature_column_value[0]) -
                            col(column_name)) /
                           col(column_name))
    update_list_features("numerical", [ratio_column_name])
    return data


def update_list_features(list_type, list_column_name):
    """Update adding name features to the list of features

    Args:
        list_type: categorical or numerical.
        list_column_name: list of features name to be added.
            If is a unique value, need be as list as well.
    """
    utils.save_log('{0} :: {1}'.format(
        update_list_features.__module__,
        update_list_features.__name__))

    for column_name in list_column_name:
        if list_type == 'categorical':
            categorical_features_list.append(column_name)
        if list_type == 'numerical':
            numerical_features_list.append(column_name)
        features_list.append(column_name)


def update_features_dims(data):
    """Update list of features and this dimension to use in SMOTENC

    Args:
        data: dataframe
    """
    utils.save_log('{0} :: {1}'.format(
        update_features_dims.__module__,
        update_features_dims.__name__))

    return [data[features_list].columns.get_loc(i)
            for i in categorical_features_list]
