import sys
import fire

import models
import preprocess
import utils as ut
import visualization

def train(**kwargs):
    """In this function, It's followed this pipeline:
    - Load input data frame (training)
    - pre-processing the data
     handle features and train model.
    $ python main.py train \
    --input_train_file ../data/xente_fraud_detection_train.csv \
    --output_balanced_train_x_file ../data/balanced_train_x.csv \
    --output_balanced_train_y_file ../data/balanced_train_y.csv \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv
    """
    ut.save_log('{0} :: {1}'.format(train.__module__,
                                    train.__name__))

    training_data = ut.read_data(kwargs['input_train_file'])

    if training_data:
        ut.save_log(train.__name__ + ' :: Input Data Not Found')
        sys.exit()

    training_data = preprocess.generate_new_features(training_data)
    training_data = models.identify_outliers(training_data)
    visualization.plot_heatmap(training_data)

    x_training_data, x_validation_data, y_training_data, y_validation_data = \
        ut.split_training_and_validation(training_data[ut.all_features],
                                         training_data[ut.label],
                                         kwargs['output_valid_x_file'],
                                         kwargs['output_valid_y_file'])

    x_training_data_balanced, y_training_data_balanced = \
        models.balance_data_set(x_training_data,
                                y_validation_data,
                                ut.categorical_features_dims,
                                kwargs['output_balanced_train_x_file'],
                                kwargs['output_balanced_train_y_file'])

    models.predict_frauds(x_training_data_balanced,
                          y_training_data_balanced,
                          ut.categorical_features_list)

    visualization.plot_feature_importance()

    print('------------ Finish Train ------------')


def validate(**kwargs):
    """Load previously trained model and validate the results.
    Execute:
    $ python main.py validate \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv
    --output_valid_result_file ../data/valid_result.csv
    """
    ut.save_log(validate.__name__ + ' :: ...Init...')
    hdl.read_validation_data(**kwargs)
    hdl.evaluate_model('VALID')
    hdl.export_data_valid_result(**kwargs)
    print('------------ Finish Validation ------------')


def test(**kwargs):
    """Load previously trained models and test it.
    Execute:
    $ python main.py test \
    --input_test_file ../data/xente_fraud_detection_test.csv \
    --output_test_result_file ../data/xente_output_final.txt
    """
    ut.save_log(test.__name__ + ' :: ...Init Test...')
    testing_data = ut.read_data(kwargs['input_test_file'])
    if testing_data:
        ut.save_log(test.__name__ + ' :: Input Data Not Found')
        sys.exit()

    hdl.handle_data_test()
    hdl.evaluate_model('TEST')
    hdl.export_data_test_result(**kwargs)
    print('------------ Finish Test ------------')


# Run all pipeline sequentially
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
    ut.save_log(run.__name__ + ' :: args: {}\n'.format(kwargs))

    # train catboost model
    train(**kwargs)
    # validate catboost model
    validate(**kwargs)
    # test catboost model in real scenario
    test(**kwargs)

    hdl.outside_log(run.__name__, '...Finish...')


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
