import models
import utils
import features_engineering


def get_percentage_of_fraudulent_transactions(data):
    """Compute the proportion of fraudulent transactions on data.

    Args:
        data (Pandas dataframe): a matrix dataframe

    Returns:
        Percentage of fraud into dataframe.
    """
    utils.save_log('{0} :: {1}'.format(
        get_percentage_of_fraudulent_transactions.__module__,
        get_percentage_of_fraudulent_transactions.__name__))

    if features_engineering.target_label in data.columns:
        features_engineering.fraudulent_percentage = \
            data.filter('FraudResult==1').count() / data.count()
    return features_engineering.fraudulent_percentage


def identify_outliers(data):
    """Create outlier detectors, train it with dataframe and append
        on data these features.

    Args:
        data (Pandas dataframe): a matrix dataframe
    Returns:
        data: Dataframe with outliers columns and sum of them
    """
    utils.save_log('{0} :: {1}'.format(
        identify_outliers.__module__,
        identify_outliers.__name__))

    percent_fraudulent_transactions = \
        get_percentage_of_fraudulent_transactions(data)

    data = data.toPandas()

    data = outliers_with_isolation_forest(
        data,
        features_engineering.numerical_features_list,
        features_engineering.target_label,
        percent_fraudulent_transactions)

    data = outliers_with_knn(
        data,
        features_engineering.numerical_features_list,
        features_engineering.target_label,
        percent_fraudulent_transactions)

    data = outliers_with_lscp(
        data,
        features_engineering.numerical_features_list,
        features_engineering.target_label,
        percent_fraudulent_transactions)

    data['SumOfOutliers'] = \
        data['IsolationForest'] + data['LSCP'] + data['KNN']

    features_engineering.update_list_features('categorical',
                                              ['IsolationForest',
                                               'LSCP',
                                               'KNN',
                                               'SumOfOutliers'])

    return data


def outliers_with_isolation_forest(data,
                                   features_columns_list,
                                   label_column: None,
                                   percentage_of_outliers: None):
    """Usage of Isolation Forest model to predict outliers into the data

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of column names (list of features)
        label_column: target column name
        percentage_of_outliers: percentage of false itens (fraud into data)

     Returns:
        data: dataframe with Isolation Forest outlier column
    """
    utils.save_log('{0} :: {1}'.format(
        outliers_with_isolation_forest.__module__,
        outliers_with_isolation_forest.__name__))

    if label_column is not None:
        models.isolation_forest.train(data,
                                      features_columns_list,
                                      label_column,
                                      percentage_of_outliers)

        predictions = \
            models.isolation_forest.predict(data[features_columns_list])
    else:
        predictions = models.isolation_forest.predict(data)

    data['IsolationForest'] = \
        models.isolation_forest.normalize_vector(predictions)

    return data


def outliers_with_lscp(data,
                       features_columns_list,
                       label_column: None,
                       percentage_of_outliers: None):
    """Usage of LSCP model to predict outliers into the data

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of column names (list of features)
        label_column: target column name
        percentage_of_outliers: percentage of false itens (fraud into data)

     Returns:
        data: dataframe with LSCP outlier column
    """
    utils.save_log('{0} :: {1}'.format(
        outliers_with_lscp.__module__,
        outliers_with_lscp.__name__))

    if label_column is not None:
        models.lscp.train(data,
                          features_columns_list,
                          label_column,
                          percentage_of_outliers)

        predictions = models.lscp.predict(data[features_columns_list])
    else:
        predictions = models.lscp.predict(data)

    data['LSCP'] = predictions

    return data


def outliers_with_knn(data,
                      features_columns_list,
                      label_column: None,
                      percentage_of_outliers: None):
    """Usage of KNN model to predict outliers into the data

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of column names (list of features)
        label_column: target column name
        percentage_of_outliers: percentage of false itens (fraud into data)

     Returns:
        data: dataframe with KNN outlier column
    """
    utils.save_log('{0} :: {1}'.format(
        outliers_with_knn.__module__,
        outliers_with_knn.__name__))

    if label_column is not None:
        models.knn.train(data,
                         features_columns_list,
                         label_column,
                         percentage_of_outliers)
        predictions = models.knn.predict(data[features_columns_list])
    else:
        predictions = models.knn.predict(data)

    data['KNN'] = predictions

    return data
