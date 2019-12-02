import fire
import config as cfg
import models
import pandas as pd
import preprocessing as proc
import visualization as vis
import sys
from fraud_detection import config  # noqa


def features(**kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

    NOTE
    ----
    """
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")
    cfg.data_train = proc.read_data(kwargs['train_file_name'])
    cfg.contamination_level = (cfg.data_train.filter('FraudResult==1').count()) / (cfg.data_train.count())
    cfg.data_train = proc.generate_new_features(cfg.data_train)
    cfg.data_train = proc.clean_data(cfg.data_train)

    proc.separate_variables(cfg.data_train)  # split data in train and test

    models.train_isolation_forest()
    models.train_LSCP()
    models.train_KNN()
    cfg.x_train[cfg.COUNT_COLUMN_NAME] = (cfg.x_train.IsolationForest + cfg.x_train.LSCP + cfg.x_train.KNN)

    proc.add_features()
    proc.balance_data()

    cfg.x_train_balanced.to_csv(kwargs['output_x_file_name'], index=False)
    cfg.y_train_balanced.to_csv(kwargs['output_y_file_name'], index=False)

    vis.plot_distribution()
    vis.plot_heatmap()


def train(**kwargs):
    """ Train a new model with data created.
    ----
    NOTE
    x_file_name

    """
    print("==> TRAINING YOUR MODEL!")
    cfg.x_train_balanced = pd.read_csv(kwargs['x_file_name'])
    cfg.y_train_balanced = pd.read_csv(kwargs['y_file_name'])
    models.train_cat_boost()
    vis.generate_explanations()


def validate(**kwargs):
    """Validate:

    NOTE
    ----

    """
    print("==> PREDICT MODEL PERFORMANCE")
    cfg.data_test = proc.read_data(kwargs['test_file_name'])
    cfg.data_test = proc.generate_new_features(cfg.data_test)
    cfg.data_test = proc.clean_data(cfg.data_test)
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
    proc.add_features(cfg.x_train_balanced)
    train()


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    run
    --train_file_name ../data/xente_fraud_detection_train.csv
    --test_file_name ../data/xente_fraud_detection_test.csv
    --output_x_file_name ../data/x_balanced_data.csv
    --output_y_file_name ../data/y_balanced_data.csv
    """
    print("Args: {}".format(kwargs))
    features(**kwargs)  # read data set and generate new features
    train()  # training model and save to filesystem
    validate(**kwargs)
    print("Everything is ok.")


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()

