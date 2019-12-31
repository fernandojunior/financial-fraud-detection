from pyod.models.lscp import LSCP
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.lof import LOF
from pyod.models.cblof import CBLOF
import pickle
import os
import numpy

import config
import utils

numpy.seterr(all='ignore')


def train(data,
          features_columns_list,
          label_column,
          percentage_of_outliers,
          output_file_name='../data/model_lscp'):
    """Fit the LSCP model using the training data.
        The model weights are saved in output file.

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of columns to use in the train
        label_column: column name fraud identification
        percentage_of_outliers: percentage of fraud on data
        output_file_name: output file name to export IF model

    Returns:
        model: LSCP model
    """
    utils.save_log('{0} :: {1}'.format(
        train.__module__,
        train.__name__))

    if os.path.isfile(output_file_name):
        utils.save_log('Loading LSCP model.')
        with open(output_file_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model

    model = create_model(percentage_of_outliers=percentage_of_outliers)
    model.fit(data[features_columns_list], data[label_column])

    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)

    return model


def create_model(percentage_of_outliers=0.002):
    """Create a LSCP model.

    Args:
        percentage_of_outliers: percentage of fraud on data

    Returns:
        model: LSCP model
    """
    utils.save_log('{0} :: {1}'.format(
        create_model.__module__,
        create_model.__name__))

    bagging_model = \
        get_model_bagging(percentage_of_outliers=percentage_of_outliers)

    lof_model = \
        get_model_lof(percentage_of_outliers=percentage_of_outliers)

    cblof_model = \
        get_model_cblof(percentage_of_outliers=percentage_of_outliers)

    list_of_detectors = [bagging_model, lof_model, cblof_model]
    model = LSCP(detector_list=list_of_detectors,
                 contamination=percentage_of_outliers)

    #model = get_model_cblof(percentage_of_outliers=percentage_of_outliers)

    return model


def get_model_bagging(percentage_of_outliers=0.002,
                      num_estimators=2,
                      combination='max'):
    """Create a Feature Bagging model.

    Args:
        percentage_of_outliers: percentage of fraud on data
        num_estimators: number of base estimators in the ensemble.
        combination: if ‘average’: take the average of all detectors
                     if ‘max’: take the maximum scores of all detectors

    Returns:
        model: Feature Bagging model
    """
    utils.save_log('{0} :: {1}'.format(
        get_model_bagging.__module__,
        get_model_bagging.__name__))

    model = FeatureBagging(contamination=percentage_of_outliers,
                           n_estimators=num_estimators,
                           combination=combination,
                           random_state=config.random_seed,
                           n_jobs=config.num_jobs)

    return model


def get_model_lof(percentage_of_outliers=0.002, num_neighbors=2):
    """Create a LOF model.

    Args:
        percentage_of_outliers: percentage of fraud on data
        num_neighbors: number of neighbors for kneighbors queries

    Returns:
        model: LOF model
    """
    utils.save_log('{0} :: {1}'.format(
        get_model_lof.__module__,
        get_model_lof.__name__))

    model = LOF(contamination=percentage_of_outliers,
                n_neighbors=num_neighbors,
                n_jobs=config.num_jobs)

    return model


def get_model_cblof(percentage_of_outliers=0.002, num_clusters=2):
    """Create a CBLOF model.

    Args:
        percentage_of_outliers: percentage of fraud on data
        num_clusters: number of clusters to form as well as the
                            number of centroids to generate

    Returns:
        model: CBLOF model
    """
    utils.save_log('{0} :: {1}'.format(
        get_model_cblof.__module__,
        get_model_cblof.__name__))

    model = CBLOF(contamination=percentage_of_outliers,
                  n_clusters=num_clusters,
                  random_state=config.random_seed,
                  n_jobs=config.num_jobs)

    return model


def predict(data, input_file_name='../data/model_lscp'):
    """Generate predictions using the LSCP model.

    Args:
        data (Pandas dataframe): a matrix dataframe
        input_file_name: input file name of LSCP model

    Returns:
        predictions: Model outcomes (predictions)
    """
    utils.save_log('{0} :: {1}'.format(
        predict.__module__,
        predict.__name__))

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    predictions = model.predict(data)

    return predictions
