import fire
import config as cfg
import models
import preprocessing as proc
import visualization as vis
from fraud_detection import config  # noqa


def features(**kwargs):
    """Function that will generate the dataset for your model. It can
    be the target population, training or validation dataset. You can
    do in this step as well do the task of Feature Engineering.

    NOTE
    ----
    config.data_path: workspace/data

    You should use workspace/data to put data to working on.  Let's say
    you have workspace/data/iris.csv, which you downloaded from:
    https://archive.ics.uci.edu/ml/datasets/iris. You will generate
    the following:

    + workspace/data/test.csv
    + workspace/data/train.csv
    + workspace/data/validation.csv
    + other files

    With these files you can train your model!
    """
    print("==> GENERATING DATASETS FOR TRAINING YOUR MODEL")
    data = proc.read_data(kwargs['input'])
    data = proc.generate_new_features(data)
    data = proc.clean_data(data)
    return data


def visualizations(data):
    """
    :param data:
    :return:
    """
    print("==> GENERATING VISUALIZATIONS FOR TRAINING YOUR MODEL")
    vis.plot_heatmap(data)


def train(data):
    """Function that will run your model, be it a NN, Composite indicator
    or a Decision tree, you name it.

    NOTE
    ----
    config.models_path: workspace/models
    config.data_path: workspace/data

    As convention you should use workspace/data to read your dataset,
    which was build from generate() step. You should save your model
    binary into workspace/models directory.

    source-code example:
    https://github.com/davified/clean-code-ml/blob/master/docs/functions.md
    how to call different models in the same function.
    """
    print("==> TRAINING YOUR MODEL!")
    proc.separate_variables(data)  # split data in train and test
    x_train = models.train_isolation_forest()
    x_train = models.train_LSCP(x_train)
    x_train = models.train_KNN(x_train)
    x_train = proc.add_features(x_train)
    x_train_bal, y_train_bal = proc.balance_data(x_train)
    return x_train


def metadata(**kwargs):
    """Generate metadata for model governance using testing!

    NOTE
    ----
    workspace_path: config.workspace_path

    In this section you should save your performance model,
    like metrics, maybe confusion matrix, source of the data,
    the date of training and other useful stuff.

    You can save like as workspace/performance.json:

    {
       'name': 'My Super Nifty Model',
       'metrics': {
           'accuracy': 0.99,
           'f1': 0.99,
           'recall': 0.99,
        },
       'source': 'https://archive.ics.uci.edu/ml/datasets/iris'
    }
    """
    print("==> TESTING MODEL PERFORMANCE AND GENERATING METADATA")


def predict(input_data):
    """Predict: load the trained model and score input_data

    NOTE
    ----
    As convention you should use predict/ directory
    to do experiments, like predict/input.csv.
    """
    print("==> PREDICT DATASET {}".format(input_data))


# Run all pipeline sequentially
def run(**kwargs):
    """Run the complete pipeline of the model.
    """
    print("Args: {}".format(kwargs))

    data = features(**kwargs)  # read dataset and generate new features
    visualizations(data)  # generate visualization above the data
    train(data)  # training model and save to filesystem
    #metadata(**kwargs)  # performance report
    print("Everything is ok.")


def cli():
    """Caller of the fire cli"""
    return fire.Fire()


if __name__ == '__main__':
    cli()
