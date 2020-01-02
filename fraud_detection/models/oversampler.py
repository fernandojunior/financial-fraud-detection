import pandas
from imblearn.over_sampling import SMOTENC

import utils
import config
import features_engineering


def smotenc_over_sampler(X_data,
                         y_data,
                         categorical_features_dims):
    """Generate oversampling for training data set using SMOTENC technique.

    Args:
        X_data (pandas data frame):
        y_data (pandas vector):
        categorical_features_dims (list):

    Returns:
        X and Y datasets balanced
    """
    utils.save_log('{0} :: {1}'.format(
        smotenc_over_sampler.__module__,
        smotenc_over_sampler.__name__))

    model = SMOTENC(categorical_features=categorical_features_dims,
                    random_state=config.random_seed,
                    n_jobs=config.num_jobs)

    X, y = model.fit_resample(X_data, y_data)

    X_smotenc = pandas.DataFrame(X,
                                 columns=features_engineering.features_list)
    y_smotenc = pandas.DataFrame(y,
                                 columns=[features_engineering.target_label])

    return X_smotenc, y_smotenc


def balance_data_set(X_data,
                     y_data,
                     categorical_features_dims):
    """Usage of KNN model to predict outliers into the data

    Args:
        X_data: a matrix dataframe
        y_data: list of column names (list of features)
        categorical_features_dims: target column name

     Returns:
        Dataframe with KNN outlier column
    """
    utils.save_log('{0} :: {1}'.format(
        balance_data_set.__module__,
        balance_data_set.__name__))

    X_data_oversampled, y_data_oversampled = \
        smotenc_over_sampler(X_data,
                             y_data,
                             categorical_features_dims)

    X_data_oversampled = pandas.DataFrame(X_data_oversampled)
    y_data_oversampled = pandas.DataFrame(y_data_oversampled)

    return X_data_oversampled, y_data_oversampled
