import isolation_forest
import lscp
import knn
import oversampler
import utils as ut


def balance_data_set(x_data_set,
                     y_data_set,
                     categorical_features_dims,
                     output_x_file_name,
                     output_y_file_name):
    """
    Args:
        - x_data_set (pandas data frame):
        - y_data_set (pandas data frame):
    """
    ut.save_log('{0} :: {1}'.format(balance_data_set.__module__,
                                    balance_data_set.__name__))

    x_data_set_oversampled, y_data_set_oversampled = \
        oversampler.smotenc_over_sampler(x_data_set,
                                         y_data_set,
                                         categorical_features_dims)

    x_data_set_oversampled.to_csv(output_x_file_name, index=False)

    y_data_set_oversampled.to_csv(output_y_file_name, index=False)

    return x_data_set_oversampled, y_data_set_oversampled


def get_percentage_of_fraudulent_transactions(data_set):
    """This function will compute the proportion of fraudulent
    transactions in training data.
    """
    ut.save_log('{0} :: {1}'.format(get_percentage_of_fraudulent_transactions.__module__,
                                    get_percentage_of_fraudulent_transactions.__name__))

    return data_set.filter('FraudResult==1').count() / data_set.count()


def identify_outliers(data_set):
    """

    """
    ut.save_log('{0} :: {1}'.format(identify_outliers.__module__,
                                    identify_outliers.__name__))

    percentage_of_fraudulent_transactions = \
        get_percentage_of_fraudulent_transactions(data_set)
    # The scikit-learn and PYOD functions require pandas or numpy data frames
    data_set = data_set.to_Pandas()
    data_set = identify_outliers_with_isolation_forest(data_set,
                                                       percentage_of_fraudulent_transactions)
    data_set = identify_outliers_with_lscp(data_set,
                                           percentage_of_fraudulent_transactions)
    data_set = identify_outliers_with_knn(data_set,
                                          percentage_of_fraudulent_transactions)
    data_set[ut.outliers_column_name] = (data_set[ut.isolation_forest_column_name] +
                                         data_set[ut.lscp_column_name] +
                                         data_set[ut.knn_column_name])
    ut.update_features_list(data_set)
    return data_set


def identify_outliers_with_isolation_forest(data_set,
                                            percentage_of_outliers):
    """
    """
    ut.save_log('{0} :: {1}'.format(identify_outliers_with_isolation_forest.__module__,
                                    identify_outliers_with_isolation_forest.__name__))

    isolation_forest.train_isolation_forest(data_set[ut.NUMERICAL_FEATURES],
                                            data_set[ut.LABEL],
                                            percentage_of_outliers)
    predictions = isolation_forest.predict_isolation_forest(data_set[ut.NUMERICAL_FEATURES])
    data_set[ut.isolation_forest_column_name] = predictions
    return data_set


def identify_outliers_with_lscp(data_set,
                                percentage_of_outliers):
    """
    """
    ut.save_log('{0} :: {1}'.format(identify_outliers_with_lscp.__module__,
                                    identify_outliers_with_lscp.__name__))

    lscp.train_lscp(data_set[ut.NUMERICAL_FEATURES],
                    data_set[ut.LABEL],
                    percentage_of_outliers)
    predictions = lscp.predict_lscp(data_set[ut.NUMERICAL_FEATURES])
    data_set[ut.lscp_column_name] = predictions
    return data_set


def identify_outliers_with_knn(data_set,
                               percentage_of_outliers):
    """
    """
    ut.save_log('{0} :: {1}'.format(identify_outliers_with_knn.__module__,
                                    identify_outliers_with_knn.__name__))

    knn.train_knn(data_set[ut.NUMERICAL_FEATURES],
                  data_set[ut.LABEL],
                  percentage_of_outliers)
    predictions = knn.predict_knn(data_set[ut.NUMERICAL_FEATURES])
    data_set[ut.knn_column_name] = predictions
    return data_set
