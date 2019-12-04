import fire
import pandas as pd

import models
import config as cfg
import handler as hdl
import visualization as vis

def extract_data(**kwargs):
    """Function that will generate the dataset for your model.
    It can be the target population, training or validation dataset.
    """
    hdl.extract_data(**kwargs)

def handle_data(**kwargs):
    hdl.outside_log(handle_data.__module__,
                    handle_data.__name__)
    """ Feature Engineering task.
    """
    hdl.get_contamination(**kwargs)
    hdl.create_features(**kwargs)
    hdl.remove_features(**kwargs)
    vis.plot_heatmap()

    hdl.split_dataset_train(**kwargs)

    #outliers--
    models.train_isolation_forest()
    models.predict_isolation_forest()
    models.train_LSCP()
    models.predict_LSCP()
    models.train_KNN()
    models.predict_KNN()
    hdl.createOutlierFeatures(**kwargs)
    #----------

    hdl.balance_oversampling(**kwargs)

    vis.plot_target_distribution()


def train(**kwargs):
    """ Train a new model with data created.
    ----
    NOTE
    x_file_name

    """
    print("==> TRAINING YOUR MODEL!")
    cfg.x_train_balanced = pd.read_csv(kwargs['x_file_name'])
    cfg.y_train_balanced = pd.read_csv(kwargs['y_file_name'])
    mdl.train_cat_boost()
    vis.generate_explanations()


def validate(**kwargs):
    """Validate:

    NOTE
    ----

    """
    print("==> PREDICT MODEL PERFORMANCE")
    cfg.data_test = hdl.read_data(kwargs['test_file_name'])
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

    extract_data(**kwargs) # read dataset
    if hdl.is_missing_data:
        handle_data(**kwargs) # handle dataset
    #train()  # training model and save to filesystem
    #validate(**kwargs)

    hdl.outside_log(run.__name__, '...Finish...')

def cli():
    """Caller of the fire cli"""
    return fire.Fire()

if __name__ == '__main__':
    cli()
