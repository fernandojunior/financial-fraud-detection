from pyod.models.knn import KNN
import utils as ut
import pickle
import os
import config as cfg


def train(data,
          features_columns_list,
          label_column,
          percentage_of_outliers,
          output_file_name='../data/model_knn'):
    """Fit the KNN model using the training data.
        The model weights are saved in output file.

    Args:
        data (Pandas dataframe): a matrix dataframe
        features_columns_list: list of columns to use in the train
        label_column: column name fraud identification
        percentage_of_outliers: percentage of fraud on data
        output_file_name: output file name to export IF model

    Returns:
        model: KNN model
    """
    ut.save_log(f'{train.__module__} :: '
                f'{train.__name__}')

    if os.path.isfile(output_file_name):
        ut.save_log('Loading KNN model.')
        with open(output_file_name, 'rb') as pickle_file:
            model = pickle.load(pickle_file)
        return model

    model = create_model(percentage_of_outliers=percentage_of_outliers)
    model.fit(data[features_columns_list], data[label_column])

    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)

    return model


def create_model(percentage_of_outliers=0.002,
                 num_neighbors=2,
                 method='largest'):
    """Create a KNN model.

    Args:
        percentage_of_outliers: percentage of fraud on data
        num_neighbors: number of neighbors for kneighbors queries
        method: ’largest’: distance to the kth neighbor as the outlier score
                ’mean’: average of all k neighbors as the outlier score
                ’median’: median of the distance to k neighbors
                            as the outlier score

    Returns:
        model: Isolation Forest model
    """
    ut.save_log(f'{create_model.__module__} :: '
                f'{create_model.__name__}')

    model = KNN(contamination=percentage_of_outliers,
                n_neighbors=num_neighbors,
                method=method,
                n_jobs=cfg.num_jobs)

    return model


def predict(data, input_file_name='../data/model_knn'):
    """Generate predictions using the KNN model.

    Args:
        data (Pandas dataframe): a matrix dataframe
        input_file_name: input file name of KNN model

    Returns:
        predictions: Model outcomes (predictions)
    """
    ut.save_log(f'{predict.__module__} :: '
                f'{predict.__name__}')

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    predictions = model.predict(data)

    return predictions
