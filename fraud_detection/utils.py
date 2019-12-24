from sklearn.metrics import precision_recall_fscore_support as score
from pyspark.sql import SparkSession
import findspark
import sklearn.model_selection as sklearn
import logging
logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(filename)s :: '
                           '%(funcName)s\n\t%(message)s')
findspark.init()
SPARK = SparkSession.builder.appName('Xente').getOrCreate()

categorical_features_dims = 0
cat_boost_column_name = 'CatBoost'

label = 'FraudResult'
value_column_name = 'Value'
isolation_forest_column_name = 'IsolationForest'
lscp_column_name = 'LSCP'
knn_column_name = 'KNN'
number_of_outliers_column_name = 'SumOfOutliers'

DEFAULT_OUTLIER_PERCENTAGE = 0.00201752001839811
fraudulent_percentage = DEFAULT_OUTLIER_PERCENTAGE

mapping_types = {'ProviderId': 'object', 'ProductId': 'object',
                 'TransactionId': 'object', 'BatchId': 'object',
                 'ProductCategory': 'object', 'ChannelId': 'object',
                 'PricingStrategy': 'int64', 'Value': 'float64',
                 'Operation': 'float64', 'TransactionHour': 'float64',
                 'DayOfWeek': 'float64', 'WeekOfYear': 'float64',
                 'RatioValueSpentByWeek': 'float64',
                 'RatioValueSpentByDayOfWeek': 'float64',
                 'RatioOfAverageValuePerProductId': 'float64',
                 'RatioOfAverageValuePerProviderId': 'float64',
                 'IsolationForest': 'int64',
                 'LSCP': 'int64', 'KNN': 'int64', 'SumOfOutliers': 'int64'}

all_features_list = ['ProviderId', 'ProductId', 'TransactionId', 'BatchId',
                     'ProductCategory', 'ChannelId', 'PricingStrategy',
                     'Value', 'Operation', 'TransactionHour', 'DayOfWeek',
                     'WeekOfYear', 'RatioValueSpentByWeek',
                     'RatioValueSpentByDayOfWeek',
                     'RatioOfAverageValuePerProductId',
                     'RatioOfAverageValuePerProviderId']

numerical_features_list = ['Value', 'Operation', 'ValueStrategy',
                           'TransactionHour', 'DayOfWeek', 'WeekOfYear',
                           'RatioValueSpentByWeek',
                           'RatioValueSpentByDayOfWeek',
                           'RatioValueSpentByDayOfYear',
                           'AverageValuePerProductId',
                           'AverageValuePerProviderId',
                           'RatioOfAverageValuePerProductId',
                           'RatioOfAverageValuePerProviderId']

categorical_features_list = ['ProviderId', 'ProductId', 'TransactionId',
                             'BatchId', 'ProductCategory', 'ChannelId',
                             'PricingStrategy']

label_name = 'FraudResult'


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
    save_log('{0} :: {1}'.format(update_features_list.__module__,
                                 update_features_list.__name__))

    global all_features_list
    global categorical_features_list
    global categorical_features_dims

    new_features_list = [isolation_forest_column_name,
                         lscp_column_name,
                         knn_column_name,
                         number_of_outliers_column_name]

    all_features_list += new_features_list
    categorical_features_list += new_features_list
    categorical_features_dims = \
        [data_set[all_features_list].columns.get_loc(i)
         for i in categorical_features_list]


def save_data_in_disk(x_validation_data,
                      y_validation_data,
                      predictions,
                      output_file_name):
    """
    Args:
        -
    """
    save_log('{0} :: {1}'.format(save_data_in_disk.__module__,
                                 save_data_in_disk.__name__))

    data = x_validation_data
    data[cat_boost_column_name] = predictions
    data[label_name] = y_validation_data

    data.to_csv(output_file_name,
                index=None,
                header=True)


def save_performance_in_disk(y_label,
                             y_predictions,
                             depth_tree=5,
                             learning_rate=0.1,
                             regularization_l2=2,
                             output_file='../data/catBoost_model_result.txt'):
    """
    Args:
    """
    save_log('{0} :: {1}'.format(save_performance_in_disk.__module__,
                                 save_performance_in_disk.__name__))

    precision, recall, f_score, _ = score(y_label, y_predictions)

    output_parser = open(output_file, 'w')
    output_parser.write('LABELS\t\tFraudResult\t\t\t\t | \tCatBoost\n')
    output_parser.write('------------------------------------------\n')
    output_parser.write('precision: \t{}\t\t | \t{}\n'.
                        format(precision[0], precision[1]))
    output_parser.write('recall: \t\t{}\t\t | \t{}\n'.
                        format(recall[0], recall[1]))
    output_parser.write('f-score: \t\t{}\t\t | \t{}\n'.
                        format(f_score[0], f_score[1]))
    output_parser.write('------------------------------------------\n')
    output_parser.write('CAT-BOOST CONFIGURATION--------------------\n')
    output_parser.write('depth: {} - LR {} - L2: {}\n'.
                        format(depth_tree,
                               learning_rate,
                               regularization_l2))
    output_parser.close()


def save_zindi_predictions(list_of_transactions_id,
                           list_of_predicted_classes,
                           output_file_name):
    """
    Args:

    """
    file = open(output_file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for transaction_id, value in \
            zip(list_of_transactions_id, list_of_predicted_classes):
        file.write('{0},{1}\n'.format(transaction_id, int(value)))
    file.close()


def normalize_vector(vector):
    """
    Input: [-1, 1]
    Output: [1, 0]

    Args:
        - vector (int):
    """
    return ((vector * -1) + 1) / 2
