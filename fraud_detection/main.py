import sys
import fire
from models import cat_boost, oversampler

import outlier_detector
import features_engineering
import utils
import visualization


def train(**kwargs):
    """Handle features and train model.

    Execute:
    $ python main.py train \
    --input_train_file ../data/xente_fraud_detection_train.csv \
    --output_balanced_train_x_file ../data/balanced_train_x.csv \
    --output_balanced_train_y_file ../data/balanced_train_y.csv \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv
    """
    utils.save_log('{0} :: {1}'.format(
        train.__module__,
        train.__name__))

    training_data = utils.read_data(kwargs['input_train_file'])

    if not training_data:
        utils.save_log(f'{train.__name__} :: ' +
                       'Input Data Not Found')
        sys.exit()

    training_data = features_engineering.generate_new_features(training_data)

    training_data = outlier_detector.identify_outliers(training_data)

    visualization.plot_heatmap(training_data)

    features_engineering.categorical_features_dims = \
        features_engineering.update_features_dims(training_data)

    utils.export_pandas_columns_to_txt(
        training_data[features_engineering.features_list])

    X_train, X_valid, y_train, y_valid = \
        utils.split_data_train_valid(
            training_data[features_engineering.features_list],
            training_data[features_engineering.target_label],
            test_proportion=0.3)

    X_train_balanced, y_train_balanced = \
        oversampler.balance_data_set(
            X_train[features_engineering.features_list],
            y_train,
            features_engineering.categorical_features_dims)

    utils.export_pandas_dataframe_to_csv(
        X_data=X_valid[features_engineering.features_list],
        y_data=y_valid,
        x_name_file=kwargs['output_valid_x_file'],
        y_name_file=kwargs['output_valid_y_file'])

    utils.export_pandas_dataframe_to_csv(
        X_data=X_train_balanced[features_engineering.features_list],
        y_data=y_train_balanced,
        x_name_file=kwargs['output_balanced_train_x_file'],
        y_name_file=kwargs['output_balanced_train_y_file'])

    cat_boost_model = \
        cat_boost.train(X_train_balanced[features_engineering.features_list],
                        y_train_balanced,
                        features_engineering.categorical_features_list)

    visualization.plot_feature_importance(
        cat_boost_model,
        X_train_balanced,
        y_train_balanced,
        features_engineering.categorical_features_list)

    print('------------ Finish Train ------------')


def validation(**kwargs):
    """Load previously trained model and validate the results.

    Execute:
    $ python main.py validation \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv \
    --output_valid_result_file ../data/valid_result.csv
    """
    utils.save_log('{0} :: {1}'.format(
        validation.__module__,
        validation.__name__))

    x_validation_data = utils.read_data(kwargs['output_valid_x_file'])
    y_validation_data = utils.read_data(kwargs['output_valid_y_file'])
    if not x_validation_data or not y_validation_data:
        print(x_validation_data.head(1))
        utils.save_log(f'{validation.__name__} ::'
                       ' Input Data Not Found')
        sys.exit()

    # atualizando colunas
    features_engineering.features_list = utils.import_pandas_columns_from_txt()
    x_validation_data = \
        x_validation_data[features_engineering.features_list].toPandas()

    predictions = cat_boost.predict(data=x_validation_data,
                                    y_value=y_validation_data)

    data_validated = x_validation_data
    data_validated['FraudResult'] = \
        y_validation_data.toPandas()
    data_validated['CatBoost'] = predictions

    utils.export_pandas_dataframe_to_csv(
        X_data=data_validated,
        y_data=None,
        x_name_file=kwargs['output_valid_result_file'],
        y_name_file=None)

    print('------------ Finish Validation ------------')


def test(**kwargs):
    """Load previously trained models and test it.

    Execute:
    $ python main.py test \
    --input_test_file ../data/xente_fraud_detection_test.csv \
    --output_test_result_file ../data/xente_output_final.txt
    """
    utils.save_log('{0} :: {1}'.format(
        test.__module__,
        test.__name__))

    testing_data = utils.read_data(kwargs['input_test_file'])
    if not testing_data:
        utils.save_log(f'{test.__name__} :: '
                       'Input Data Not Found')
        sys.exit()

    testing_data = features_engineering.generate_new_features(testing_data)

    testing_data = outlier_detector.identify_outliers(testing_data)
    transaction_column = testing_data['TransactionId']

    # atualizando colunas
    features_engineering.features_list = utils.import_pandas_columns_from_txt()
    testing_data = testing_data[features_engineering.features_list]

    predictions = cat_boost.predict(data=testing_data, y_value=None)

    testing_data['TransactionId'] = transaction_column
    testing_data['CatBoost'] = predictions

    utils.export_pandas_dataframe_to_csv(
        X_data=testing_data,
        y_data=None,
        x_name_file='../data/test_result.csv',
        y_name_file=None)

    utils.save_zindi_predictions(testing_data['TransactionId'],
                                 testing_data['CatBoost'],
                                 kwargs['output_test_result_file'])

    print('------------ Finish Test ------------')


def run(**kwargs):
    """To run the complete pipeline of the model.
    Execute:
    $ python main.py run \
    --input_train_file ../data/xente_fraud_detection_train.csv \
    --input_test_file ../data/xente_fraud_detection_test.csv \
    --output_balanced_train_x_file ../data/balanced_train_x.csv \
    --output_balanced_train_y_file ../data/balanced_train_y.csv \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv \
    --output_valid_result_file ../data/valid_result.csv \
    --output_test_result_file ../data/xente_output_final.txt
    """
    utils.save_log(f'{run.__name__}' + ' :: '
                                       'args: {}\n'.format(kwargs))

    train(**kwargs)
    validation(**kwargs)
    test(**kwargs)

    utils.save_log(f'{run.__name__}\n ...Finish...')


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
