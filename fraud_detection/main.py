import sys
import fire

import handler as hdl


def train(**kwargs):
    """Load Dataframe, handle features and train model.
    $ python main.py train \
    --input_train_file ../data/xente_fraud_detection_train.csv \
    --output_balanced_train_x_file ../data/balanced_train_x.csv \
    --output_balanced_train_y_file ../data/balanced_train_y.csv \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv
    """
    hdl.outside_log(train.__name__, '...Init Train...')
    if hdl.extract_data_train(**kwargs):
        hdl.outside_log(train.__name__, 'Input Data Not Found')
        sys.exit()

    hdl.handle_data_train(**kwargs)
    hdl.split_train_val(**kwargs)
    hdl.balance_oversampling(**kwargs)
    hdl.train_model()
    print('------------ Finish Train ------------')


def validate(**kwargs):
    """Load previously trained model and validate the results.
    Execute:
    $ python main.py validate \
    --output_valid_x_file ../data/valid_x.csv \
    --output_valid_y_file ../data/valid_y.csv
    --output_valid_result_file ../data/valid_result.csv
    """
    hdl.outside_log(validate.__name__, '...Init...')
    if not hdl.is_missing_file_validation:
        print('------------ No Model Trained Found ------------')
        sys.exit()

    hdl.extract_data_validation(**kwargs)
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
    hdl.outside_log(test.__name__, '...Init Test...')
    if hdl.extract_data_test(**kwargs):
        hdl.outside_log(test.__name__, 'Input Data Not Found')
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
    hdl.outside_log(run.__name__, 'args: {}\n'.format(kwargs))

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
