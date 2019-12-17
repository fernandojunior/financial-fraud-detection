from pyod.models.knn import KNN
import utils as ut
import pickle


def train_knn(x_data_set,
              y_data_set,
              percentage_of_outliers,
              output_file_name='../data/model_knn'):
    """Fit the KNN model using the training data.
    Save the weights in output file.
    """
    ut.save_log('{0} :: {1}'.format(train_knn.__module__,
                                    train_knn.__name__))

    model = get_model_knn(percentage_of_outliers)
    model.fit(x_data_set, y_data_set)

    with open(output_file_name, 'wb') as file_model:
        pickle.dump(model, file_model)


def get_model_knn(percentage_of_outliers=0.002,
                  num_neighbors=3,
                  method='mean',
                  num_jobs=12):
    """Retrieve KNN model.
    """
    ut.save_log('{0} :: {1}'.format(get_model_knn.__module__,
                                    get_model_knn.__name__))

    model = KNN(contamination=percentage_of_outliers,
                n_neighbors=num_neighbors,
                method=method,
                n_jobs=num_jobs)

    return model


def predict_knn(x_data_set,
                input_file_name='../data/model_knn'):
    """Generate predictions using the Locally Selective Combination of
    KNN model.
    This model require the previous model trained or the weights to load.
    The predictions are made using only numerical features.
    A new column is created with the predictions made by KNN model.

    Source:
    https://pyod.readthedocs.io/en/latest/_modules/pyod/models/knn.html
    """
    ut.save_log('{0} :: {1}'.format(predict_knn.__module__,
                                    predict_knn.__name__))

    with open(input_file_name, 'rb') as pickle_file:
        model = pickle.load(pickle_file)
    predictions = model.predict(x_data_set)

    return predictions
