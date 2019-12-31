from pyspark.sql import SparkSession
import findspark
import sklearn.model_selection as sklearn
import logging

import config

logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(filename)s :: '
                           '%(funcName)s\n\t%(message)s')
findspark.init()
SPARK = SparkSession.builder.appName('Xente').getOrCreate()

cat_boost_column_name = 'CatBoost'


def save_log(message):
    """Logs a message with level INFO on the root logger.

    Args:
        message (str): message describing some fact to save.
    """
    logging.info(message)


def read_data(file_name):
    """Read data frame in Spark format.

    Args:
        kwargs[input_file_name] (str): input file name for data set,
        could be a url or a path to a local file.

    Returns:
        Spark data frame: data set read.
    """
    save_log(f'{read_data.__module__} :: '
             f'{read_data.__name__}')

    data = SPARK.read.csv(file_name,
                          header=True,
                          inferSchema=True)

    return data


def split_data_train_valid(X_data, y_data, test_proportion=0.3):
    """Split data in training data and validation data

    Args:
        X_data: input data with features columns
        y_data: input data with target outcomes column
        test_proportion: proportion to split train/valid
            default 0.3-> train 70% - valid 30%

    Returns:
        Spark data frame: data set read.
    """
    save_log(f'{split_data_train_valid.__module__} :: '
             f'{split_data_train_valid.__name__}')

    X_train, X_valid, y_train, y_valid = \
        sklearn.train_test_split(X_data,
                                 y_data,
                                 test_size=test_proportion,
                                 random_state=config.random_seed)

    return X_train, X_valid, y_train, y_valid


def export_pandas_columns_to_txt(
        data,
        input_file_name='../data/features_columns.txt'):
    """ Export dataframe columns to txt file

    Args:
        - data (pandas data frame): Pandas dataframe file
    """
    save_log(f'{export_pandas_columns_to_txt.__module__} :: '
             f'{export_pandas_columns_to_txt.__name__}')

    columns = data.columns
    file = open(input_file_name, 'w')
    for col in columns:
        file.write('{0}\n'.format(col))
    file.close()


def import_pandas_columns_from_txt(file_name='../data/features_columns.txt'):
    """ Recovering dataframe columns from txt file

    Args:
        - file_name: file path to get column names
    """
    file_text = open(file_name, "r")
    array_features = []
    for line in file_text:
        array_features.append(line.split('\n')[0])
    file_text.close()
    return array_features


def export_pandas_dataframe_to_csv(X_data,
                                   y_data: None,
                                   x_name_file,
                                   y_name_file: None):
    """ Export Pandas dataframe to CSV file

    Args:
        - X_data (pandas data frame): Pandas dataframe file
        - y_data (pandas data frame): Pandas dataframe file
        - x_name_file (str): path to export the first pandas dataframe file
        - y_name_file (str): path to export the second pandas dataframe file
    """
    save_log(f'{export_pandas_dataframe_to_csv.__module__} :: '
             f'{export_pandas_dataframe_to_csv.__name__}')

    X_data.to_csv(x_name_file, index=False, header=True)
    if y_data is not None:
        y_data.to_csv(y_name_file, index=False, header=True)


def save_zindi_predictions(list_of_transactions_id,
                           list_of_predicted_classes,
                           output_file_name):
    """ Export dataframe columns to txt file

    Args:
        - data (pandas data frame): Pandas dataframe file
    """
    save_log(f'{save_zindi_predictions.__module__} :: '
             f'{save_zindi_predictions.__name__}')

    file = open(output_file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for transaction_id, value in \
            zip(list_of_transactions_id, list_of_predicted_classes):
        file.write('{0},{1}\n'.format(transaction_id, int(value)))
    file.close()
