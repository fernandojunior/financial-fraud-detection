import utils as ut
from imblearn.over_sampling import SMOTENC
import config as cfg
import features_engineering as fte
import pandas as pd


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
    ut.save_log(f'{smotenc_over_sampler.__module__} :: '
                f'{smotenc_over_sampler.__name__}')

    model = SMOTENC(categorical_features=categorical_features_dims,
                    random_state=cfg.random_seed,
                    n_jobs=cfg.num_jobs)

    X, y = model.fit_resample(X_data, y_data)

    X_smotenc = pd.DataFrame(X, columns=fte.features_list)
    y_smotenc = pd.DataFrame(y, columns=[fte.target_label])

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
    ut.save_log(f'{balance_data_set.__module__} :: '
                f'{balance_data_set.__name__}')

    '''if os.path.isfile(output_x_file_name) and \
            os.path.isfile(output_y_file_name):
        X_data_set_oversampled = pd.read_csv(output_x_file_name)
        y_data_set_oversampled = pd.read_csv(output_y_file_name)
        return X_data_set_oversampled, y_data_set_oversampled'''

    X_data_oversampled, y_data_oversampled = \
        smotenc_over_sampler(X_data,
                             y_data,
                             categorical_features_dims)

    X_data_oversampled = pd.DataFrame(X_data_oversampled,
                                      columns=fte.features_list)
    y_data_oversampled = pd.DataFrame(y_data_oversampled,
                                      columns=[fte.target_label])

    return X_data_oversampled, y_data_oversampled
