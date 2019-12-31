import os
import pickle
from sklearn.ensemble import IsolationForest

import utils
import config


def train(data,
          features_columns_list,
          label_column,
          percentage_of_outliers,
          output_file_name='../data/model_if'):
    """Fit the Isolation Forest model using the training data.
        The model weights are saved in output file.

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of columns to use in the train
        label_column: column name fraud identification
        percentage_of_outliers: percentage of fraud on data
        output_file_name: output file name to export IF model

    Returns:
        model: Isolation Forest model
    """
    utils.save_log(f'{train.__module__} :: '
                   f'{train.__name__}')

    if os.path.isfile(output_file_name):
        utils.save_log('Loading Isolation Forest model.')
        with open(output_file_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model

    model = create_model(percentage_of_outliers=percentage_of_outliers)
    model.fit(data[features_columns_list], data[label_column])

    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)

    return model


def create_model(percentage_of_outliers=0.002,
                 num_estimators=5):
    """Create a Isolation Forest model.

    Args:
        percentage_of_outliers: percentage of fraud on data
        num_estimators

    Returns:
        model: Isolation Forest model
    """
    utils.save_log(f'{create_model.__module__} :: '
                   f'{create_model.__name__}')

    model = IsolationForest(contamination=percentage_of_outliers,
                            behaviour='new',
                            n_estimators=num_estimators,
                            random_state=config.random_seed,
                            n_jobs=config.num_jobs)

    return model


def predict(data, input_file_name='../data/model_if'):
    """Generate predictions using the Isolation Forest model.

    Args:
        data (Pandas dataframe): a matrix dataframe
        input_file_name: input file name of IF model

    Returns:
        predictions: Model outcomes (predictions)
    """
    utils.save_log(f'{predict.__module__} :: '
                   f'{predict.__name__}')

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)

    predictions = model.predict(data)

    return predictions


def normalize_vector(vector):
    """Normalize the values of prediction to 0 and 1

    Args:
        vector : column predictions made by Isolation Forest

    Returns:
        vector_normalized: a column value of Isolation Forest normalized
    """
    return ((vector - 1) * -1) // 2
