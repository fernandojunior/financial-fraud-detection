from pyspark.sql.functions import (when, col, hour, dayofweek,
                                   weekofyear, dayofyear)
import utils as ut


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
F
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
    ut.save_log('{0} :: {1}'.format(
        create_feature_is_credit_or_debit.__module__,
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
    ut.save_log('{0} :: {1}'.format(
        create_feature_how_huge_is_value.__module__,
        create_feature_how_huge_is_value.__name__))

    avg_value = data.agg({ut.value_column_name: 'avg'}).collect()[0][0]
    data = \
        data.withColumn('ValueStrategy',
                        when(col(ut.value_column_name) > avg_value * 100, 3).
                        when(col(ut.value_column_name) > avg_value * 10, 2).
                        when(col(ut.value_column_name) > avg_value * 2, 1).
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
    ut.save_log('{0} :: {1}'.format(
        create_features_from_transaction_timestamp.__module__,
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
    ut.save_log('{0} :: {1}'.format(
        create_feature_based_on_spent_by_timestamp.__module__,
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
    ut.save_log('{0} :: {1}'.format(
        create_features_for_transaction_value_and_category.__module__,
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
    ut.save_log('{0} :: {1}'.format(
        create_feature_average_value_for_category.__module__,
        create_feature_average_value_for_category.__name__))

    column_name = 'avg_vl_{0}'.format(item)
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
    ut.save_log('{0} :: {1}'.format(
        create_feature_ratio_between_value_and_category.__module__,
        create_feature_ratio_between_value_and_category.__name__))

    column_name = 'avg_vl_{0}'.format(item)
    ratio_column_name = 'rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col('Value') - col(column_name)) /
                           col(column_name))
    return data
