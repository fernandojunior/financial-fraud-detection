import pickle
import os
import utils as ut
from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF


def train_lscp(data_set,
               x_columns_list,
               y_column_name,
               percentage_of_outliers,
               output_file_name='../data/model_lscp'):
    """Fit the LSCP model using the training data.
    Save the weights in output file.
    """
    ut.save_log('{0} :: {1}'.format(train_lscp.__module__,
                                    train_lscp.__name__))

    if os.path.isfile(output_file_name):
        ut.save_log('Loading LSCP model.')
        with open(output_file_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model

    model = get_model_lscp(percentage_of_outliers=percentage_of_outliers)
    model.fit(data_set[x_columns_list], data_set[y_column_name])
    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)

    return model


def get_model_lscp(percentage_of_outliers=0.002,
                   random_seed=42):
    """Retrieve the LSCP model.

    Args:
        - percentage_of_outliers (float):
    """
    ut.save_log('{0} :: {1}'.format(get_model_lscp.__module__,
                                    get_model_lscp.__name__))

    bagging_model = get_model_bagging(percentage_of_outliers)
    lof_model = get_model_lof(percentage_of_outliers)
    cblof_model = get_model_cblof(percentage_of_outliers)
    list_of_detectors = [bagging_model, lof_model, cblof_model]
    model = LSCP(detector_list=list_of_detectors,
                 contamination=percentage_of_outliers,
                 random_state=random_seed)

    return model


def get_model_bagging(percentage_of_outliers=0.002,
                      num_estimators=2,
                      combination='max',
                      random_seed=42,
                      num_jobs=8):
    """
    Source:

    Args:
        - percentage_of_outliers (float):
    """
    ut.save_log('{0} :: {1}'.format(get_model_bagging.__module__,
                                    get_model_bagging.__name__))

    model = FeatureBagging(contamination=percentage_of_outliers,
                           n_estimators=num_estimators,
                           combination=combination,
                           random_state=random_seed,
                           n_jobs=num_jobs)

    return model


def get_model_lof(percentage_of_outliers=0.002,
                  num_neighbors=2,
                  num_jobs=8):
    """

    Source:

    Args:
        - percentage_of_outliers (float):
        - num_neighbors (int):
        - num_jobs (int):
    """
    ut.save_log('{0} :: {1}'.format(get_model_lof.__module__,
                                    get_model_lof.__name__))

    model = LOF(contamination=percentage_of_outliers,
                n_neighbors=num_neighbors,
                n_jobs=num_jobs)

    return model


def get_model_cblof(percentage_of_outliers=0.002,
                    num_clusters=2,
                    random_seed=42,
                    num_jobs=8):
    """
    Source:

    Args:
        - percentage_of_outliers (float):
    """
    ut.save_log('{0} :: {1}'.format(get_model_cblof.__module__,
                                    get_model_cblof.__name__))

    model = CBLOF(contamination=percentage_of_outliers,
                  n_clusters=num_clusters,
                  random_state=random_seed,
                  n_jobs=num_jobs)

    return model


def predict_lscp(x_data_set,
                 input_file_name='../data/model_lscp'):
    """Generate predictions using the Locally Selective Combination of
    Parallel Outlier Ensembles (LSCP) model.
    This model require the previous model trained or the weights to load.
    The predictions are made using only numerical features.
    A new column is created with the predictions made by LSCP model.

    Source:
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/lscp.html

    Args:
        - percentage_of_outliers (float):
    """
    ut.save_log('{0} :: {1}'.format(predict_lscp.__module__,
                                    predict_lscp.__name__))

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    predictions = model.predict(x_data_set)

    return predictions
