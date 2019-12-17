import sklearn.model_selection as sklearn
import logging
logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(filename)s :: '
                           '%(funcName)s\n\t%(message)s')


from pyspark.sql import SparkSession
import findspark
findspark.init()
SPARK = SparkSession.builder.appName('Xente').getOrCreate()

categorical_features_dims = 0
label = ['FrauldResult']
isolation_forest_column_name = ['IsolationForest']
lscp_column_name = ['LCSP']
knn_column_name = ['Knn']
outliers_column_name = ['Sum_of_outliers']

all_features_list = ['ProviderId', 'ProductId', 'TransactionId', 'BatchId',
                     'ProductCategory', 'ChannelId', 'PricingStrategy', 'Value',
                     'Operation', 'Hour', 'DayOfWeek', 'WeekOfYear', 'Vl_per_weekYr',
                     'Vl_per_dayWk', 'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']

numerical_features_list = ['Value', 'Operation', 'Hour', 'DayOfWeek', 'WeekOfYear',
                           'Vl_per_weekYr', 'Vl_per_dayWk', 'rt_avg_vl_ProductId',
                           'rt_avg_vl_ProviderId']

categorical_features_list = ['ProviderId', 'ProductId', 'TransactionId',
                             'BatchId', 'ProductCategory', 'ChannelId',
                             'PricingStrategy']


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
    save_log('{0} :: {1}'.format(read_data.__module__,
                                 read_data.__name__))

    data = SPARK.read.csv(file_name,
                          header=True,
                          inferSchema=True)

    return data


def split_training_and_validation(x_data_set,
                                  y_data_set,
                                  output_x_validation_file_name,
                                  output_y_validation_file_name,
                                  test_proportion=0.3,
                                  random_seed=42):
    """
    Args:
        - x_data_set (pandas data frame):
        - y_data_set (pandas data frame):
        - test_proportion (float):
        - output_x_file_name (str):
        - output_y_file_name (str):
    """
    save_log('{0} :: {1}'.format(split_training_and_validation.__module__,
                                 split_training_and_validation.__name__))

    x_train, x_validation, y_train, y_validation = \
        sklearn.train_test_split(x_data_set,
                                 y_data_set,
                                 test_size=test_proportion,
                                 random_state=random_seed)

    x_validation.to_csv(output_x_validation_file_name,
                        index=None, header=True)

    y_validation.to_csv(output_y_validation_file_name,
                        index=None, header=True)

    return x_train, x_validation, y_train, y_validation


def update_categorical_features_list(content_to_be_include):
    """

    Args:
        -
    """
    save_log('{0} :: {1}'.format(update_categorical_features_list.__module__,
                                 update_categorical_features_list.__name__))

    global categorical_features_list
    categorical_features_list += content_to_be_include


def update_features_list(data_set):
    """The variables CATEGORICAL_FEATURES and ALL_FEATURES
    keep the relation of categorical features, so this
    function update these lists.
    """
    global all_features_list
    global categorical_features_list
    global categorical_features_dims

    new_features_list = [isolation_forest_column_name,
                         lscp_column_name,
                         knn_column_name,
                         outliers_column_name]

    all_features_list += new_features_list
    categorical_features_list += new_features_list
    categorical_features_dims = \
        [data_set.columns.get_loc(i) for i in categorical_features_list[:]]


def normalize_vector(vector):
    """
    Input: [-1, 1]
    Output: [1, 0]

    Args:
        - vector (int):
    """
    return ((vector * -1) + 1) / 2