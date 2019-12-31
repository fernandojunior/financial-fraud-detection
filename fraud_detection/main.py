import sys
import fire

from models import cat_boost, oversampler
import detect_outlier
import features_engineering as fte
import utils as ut
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
    ut.save_log(f'{train.__module__} :: '
                f'{train.__name__}')

    training_data = ut.read_data(kwargs['input_train_file'])

    if not training_data:
        ut.save_log(f'{train.__name__} :: '
                    'Input Data Not Found')
        sys.exit()

    training_data = fte.generate_new_features(training_data)

    training_data = detect_outlier.identify_outliers(
        training_data,
        lscp_bag_num_of_estimators=kwargs['lscp_bag_num_of_estimators'],
        lscp_lof_num_neighbors=kwargs['lscp_lof_num_neighbors'],
        lscp_cblof_num_clusters=kwargs['lscp_cblof_num_clusters'],
        knn_num_neighbors=kwargs['knn_num_neighbors'],
        knn_method=kwargs['knn_method'])

    visualization.plot_heatmap(training_data)

    fte.update_features_dims(training_data)

    ut.export_pandas_columns_to_txt(training_data[fte.features_list])

    X_train, X_valid, y_train, y_valid = \
        ut.split_data_train_valid(training_data[fte.features_list],
                                  training_data[fte.target_label],
                                  test_proportion=0.3)

    X_train_balanced, y_train_balanced = \
        oversampler.balance_data_set(X_train,
                                     y_train,
                                     fte.categorical_features_dims)

    ut.export_pandas_dataframe_to_csv(
        X_data=X_valid,
        y_data=y_valid,
        x_name_file=kwargs['output_valid_x_file'],
        y_name_file=kwargs['output_valid_y_file'])

    ut.export_pandas_dataframe_to_csv(
        X_data=X_train_balanced,
        y_data=y_train_balanced,
        x_name_file=kwargs['output_balanced_train_x_file'],
        y_name_file=kwargs['output_balanced_train_y_file'])

    cat_boost_model = \
        cat_boost.train(X_train_balanced[fte.features_list],
                        y_train_balanced,
                        fte.categorical_features_list,
                        catboost_depth=kwargs['catboost_depth'],
                        catboost_learning_rate=kwargs[
                            'catboost_learning_rate'],
                        catboost_l2_leaf_reg=kwargs['catboost_l2_leaf_reg'])

    visualization.plot_feature_importance(cat_boost_model,
                                          X_train_balanced,
                                          y_train_balanced,
                                          fte.categorical_features_list)

    print('------------ Finish Train ------------')


def validation(**kwargs):
    """Load previously trained model and validate the results.

    Execute:
    $ python main.py validation \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv \
    --output_valid_result_file ../data/valid_result.csv
    """
    ut.save_log(f'{validation.__module__} :: '
                f'{validation.__name__}')

    x_validation_data = ut.read_data(kwargs['output_valid_x_file'])
    y_validation_data = ut.read_data(kwargs['output_valid_y_file'])
    if not x_validation_data or not y_validation_data:
        print(x_validation_data.head(1))
        ut.save_log(f'{validation.__name__} :: '
                    'Input Data Not Found')
        sys.exit()

    # atualizando colunas
    fte.features_list = ut.import_pandas_columns_from_txt()
    x_validation_data = x_validation_data[fte.features_list].toPandas()

    predictions = cat_boost.predict(data=x_validation_data,
                                    y_value=y_validation_data)

    data_validated = x_validation_data
    data_validated['FraudResult'] = \
        y_validation_data.toPandas()
    data_validated['CatBoost'] = predictions

    ut.export_pandas_dataframe_to_csv(
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
    ut.save_log(f'{test.__module__} :: '
                f'{test.__name__}')

    testing_data = ut.read_data(kwargs['input_test_file'])
    if not testing_data:
        ut.save_log(f'{test.__name__} :: '
                    'Input Data Not Found')
        sys.exit()

    testing_data = fte.generate_new_features(testing_data)
    testing_data = detect_outlier.identify_outliers(testing_data)

    # atualizando colunas
    fte.features_list = ut.import_pandas_columns_from_txt()
    testing_data = testing_data[fte.features_list]

    predictions = cat_boost.predict(data=testing_data, y_value=None)

    ut.save_zindi_predictions(testing_data['TransactionId'],
                              predictions,
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
    ut.save_log(f'{run.__name__}' + ' :: '
                'args: {}\n'.format(kwargs))

    train(**kwargs)
    validation(**kwargs)
    test(**kwargs)

    ut.save_log(f'{run.__name__}\n ...Finish...')


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
