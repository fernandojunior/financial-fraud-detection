import sys
import fire
import pandas as pd

import config as cfg
import handler as hdl

def train(**kwargs):
    if hdl.extract_data(**kwargs):
        print('------------ No Data Found ------------')
        sys.exit()

    hdl.handle_data_train(**kwargs)
    hdl.balance_oversampling(**kwargs)
    hdl.train_model(**kwargs)

def validate(**kwargs):
    if not hdl.is_missing_file_validation:
        print('------------ No Model Trained Found ------------')
        sys.exit()    

    hdl.extract_data_to_validation(**kwargs)
    cfg.data_test = hdl.generate_new_features(cfg.data_test)
    cfg.data_test = hdl.clean_data(cfg.data_test)
    ### Make predictions
    ### Save in csv format


def test(**kwargs):
    """Load previously trained models and run.
    test
    --x_file_name ../data/x_balanced_data.csv
    --y_file_name ../data/y_balanced_data.csv
    --test_file_name ../data/xente_fraud_detection_test.csv
    """
    print("Args: {}".format(kwargs))
    cfg.x_train_balanced = pd.read_csv(kwargs['x_file_name'])
    cfg.y_train_balanced = pd.read_csv(kwargs['y_file_name'])
    hdl.add_features(cfg.x_train_balanced)
    train()


# Run all pipeline sequentially
def run(**kwargs):
    '''To run the complete pipeline of the model.
    Execute:
    $ python main.py run \
    --train_file_name ../data/xente_fraud_detection_train.csv \
    --test_file_name ../data/xente_fraud_detection_test.csv \
    --output_x_file_name ../data/x_balanced_data.csv \
    --output_y_file_name ../data/y_balanced_data.csv
    '''
    
    hdl.outside_log(run.__name__, '...Init...')
    hdl.outside_log(run.__name__, 'args: {}\n'.format(kwargs))

    train(**kwargs) # train catboost model

    hdl.outside_log(run.__name__, '...Finish...')

def cli():
    """Caller of the fire cli"""
    return fire.Fire()

if __name__ == '__main__':
    cli()
