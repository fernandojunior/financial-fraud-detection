import pickle
import utils as ut
from sklearn.ensemble import IsolationForest


def train_isolation_forest(x_data_set,
                           y_data_set,
                           percentage_of_outliers,
                           output_file_name='../data/model_if'):
    """Fit the Isolation Forest model using the training data.
    Save the weights in output file.

    Args:
        -
    """
    ut.save_log('{0} :: {1}'.format(train_isolation_forest.__module__,
                                    train_isolation_forest.__name__))

    model = get_isolation_forest_model(percentage_of_outliers)
    model.fit(x_data_set, y_data_set)
    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)

    return model


def get_isolation_forest_model(percentage_of_outliers=0.002,
                               behaviour='new',
                               random_seed=42,
                               num_jobs=12):
    """Retrieve the Isolation Forest model.

    Args:
        -
    """
    ut.save_log('{0} :: {1}'.format(get_isolation_forest_model.__module__,
                                    get_isolation_forest_model.__name__))

    model = IsolationForest(contamination=percentage_of_outliers,
                            behaviour=behaviour,
                            random_state=random_seed,
                            n_jobs=num_jobs)

    return model


def predict_isolation_forest(x_data_set,
                             input_file_name='../data/model_if'):
    """Generate predictions using the Isolation Forest model.
    This model require the previous model trained or the weights to load.
    The predictions are made using only numerical features.
    A new column is created with the predictions made by Isolation
    Forest model.
    The prediction are converted to keep the pattern: 0 for genuine
    transaction, and 1 for fraudulent.

    Source:
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/iforest.html
    """
    ut.save_log('{0} :: {1}'.format(predict_isolation_forest.__module__,
                                    predict_isolation_forest.__name__))

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    predictions = model.predict(x_data_set)
    predictions = ut.normalize_vector(predictions)

    return predictions