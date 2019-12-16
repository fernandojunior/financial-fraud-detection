import logging
import os.path

import pandas as pd
import sklearn.model_selection as sk_ml
from sklearn.metrics import precision_recall_fscore_support as score
from pyspark.sql.functions import (when, col, hour, dayofweek,
                                   weekofyear, dayofyear)

import config as cfg
import models
import visualization as vis

from pyspark.sql import SparkSession
import findspark
findspark.init()
SPARK = SparkSession.builder.appName('Xente').getOrCreate()

logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(filename)s :: '
                           '%(funcName)s\n\t%(message)s')


def outside_log(function_name, message):
    """Logs a message with level INFO on the root logger.

    Args:
        function_name (str): name of the function that called this method.
        message (str): message describing some fact to save.
    """
    logging.info(function_name + '.py :: ' + message)


def read_validation_data(**kwargs):
    """Read validation data.

    In this process, It is required the 'x' and 'y' data.
    They are saved in separately variables.

    Args:
        kwargs[validation_x_file_name] (str): validation vector.
        kwargs[validation_y_file_name] (str): target vector relative to X.
    """
    logging.info(extract_data_balanced.__name__)
    cfg.x_validation = pd.read_csv(kwargs['output_valid_x_file'])
    cfg.y_validation = pd.read_csv(kwargs['output_valid_y_file'])
    cfg.x_to_predict_catboost = cfg.x_validation


def read_data(file_name):
    """Read data frame in Spark format.

    Args:
        kwargs[input_file_name] (str): input file name for data set,
        could be a url or a path to a local file.

    Returns:
        Spark data frame: data set read.
    """
    data = SPARK.read.csv(file_name,
                              header=True,
                              inferSchema=True)
    return data


def pre_process_train_data(**kwargs):
    """These method is responsible for pre-processing the
    training data completely.

    Pipeline:
        - Compute_outliers_proportion: relation between
        the number of fraudulent transactions and the total
        number of transactions.
        - create_features: new features will be automatically
        generate for this data set.
        - plot_heatmap: plot a heat map visualization with
        correlation level between the features.
        -




    """
    logging.info('Handling Data Train')
    compute_fraud_proportion()
    create_features(cfg.data_train)
    vis.plot_heatmap('INIT')
    set_df_from_data_train(**kwargs)
    # outliers
    models.train_isolation_forest()
    models.predict_isolation_forest()
    models.train_lscp()
    models.predict_lscp()
    models.train_knn()
    models.predict_knn()
    create_feature_outlier_intensity()
    # get back data to x_train
    cfg.x_train = cfg.x_data_temp
    vis.plot_heatmap('OUTLIER')


def handle_data_test():
    logging.info('Handling Data Test')
    # create new features
    create_features(cfg.data_test)
    cfg.ALL_FEATURES = cfg.ALL_FEATURES_TEST
    models.predict_isolation_forest()
    models.predict_lscp()
    models.predict_knn()
    create_feature_outlier_intensity()
    cfg.x_to_predict_catboost = cfg.x_data_temp


def compute_fraud_proportion():
    """This function will compute the proportion of fraudulent
    transactions in training data.
    """
    logging.info('Computing the fraudulent proportion')
    cfg.percentage_of_fraudulent_transactions = \
        (cfg.data_train.filter('FraudResult==1').count()) / \
        (cfg.data_train.count())


def create_features(input_data):
    """Make data augmentation in input data generating new features.

    Args:
        input_data (spark data frame): this vector contains the data
        from Zindi challenge used to generate the new features.
    """
    logging.info('Creating pre-defined features')
    cfg.x_data_temp = input_data
    cfg.x_data_temp = generate_new_features(cfg.x_data_temp)
    cfg.x_data_temp = cfg.x_data_temp.toPandas()


def generate_new_features(data):
    """This function is responsible for call multiples other functions
    to create the new features:

    Pipeline:
        - create_feature_is_credit_or_debit: create a new column identifying
        the transaction, 1 for debit and -1 for credit transactions
        - create_feature_how_huge_is_value: we use the mean of transactions
        to generate new features, identifying how distant the actual
        transaction is from the average.
        - create_features_based_on_timestamp_event:
        - get_value_per_period:
        - get_features_per_value:

    Args:
        data (spark data frame): vector used to generate new features.

    Returns:
        data (spark data frame): with data augmentation.
    """
    data = create_feature_is_credit_or_debit(data)
    data = create_feature_how_huge_is_value(data)
    data = create_features_based_on_timestamp_event(data)
    data = create_feature_based_on_spent_by_timestamp(data)
    data = create_features_for_transaction_value_and_category(data)
    return data


def create_feature_is_credit_or_debit(data):
    """ This function create a new column (called Operation) that
    identify if the operation is credit (-1) or debit (1) transaction
    based on 'Amount' column.

    Args:
        data (spark data frame): input spark data frame.
    """
    data = data.withColumn("Operation",
                           when(data.Amount > 0, 1).
                           when(data.Amount < 0, -1).
                           otherwise(0))
    return data


def create_feature_how_huge_is_value(data):
    """Create a feature identifying how distant is the transaction
    value from the average.

    Distances:
        - 3: the transaction is at least 100 times bigger than average.
        - 2: the transaction is at least 10 times bigger than average.
        - 1: the transaction is at least 2 times bigger than average.
        - 0: if none of the previous conditions are satisfied.

    Args:
        data (spark data frame): input spark data frame.
    """
    avg_value = data.agg({cfg.COLUMN_VALUE: 'avg'}).collect()[0][0]
    data = data.withColumn('ValueStrategy',
                           when(col(cfg.COLUMN_VALUE) > avg_value * 100, 3).
                           when(col(cfg.COLUMN_VALUE) > avg_value * 10, 2).
                           when(col(cfg.COLUMN_VALUE) > avg_value * 2, 1).
                           otherwise(0))
    return data


def create_features_based_on_timestamp_event(data):
    """Each transaction has a specific timestamp identifying
    when this happened. So, new features are created based on
    this.

    Features:
        - Hour: hour time identifying the moment the
        transaction happened.
        - DayOfWeek: identify which day of week the
        transaction was made.
        - DayOfYear: identify which day of year the
        transaction was made.
        - WeekOfYear: identify which week of year the
        transaction was made.

    Args:
        data (spark data frame): input spark data frame.
    """
    data = data.withColumn('Hour',
                           hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek',
                           dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear',
                           dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear',
                           weekofyear(data['TransactionStartTime']))
    return data


def create_feature_based_on_spent_by_timestamp(data):
    """Ratio between value information and some timestamp
    feature created previously.

    Features:
        - Vl_per_weekYr: ration between the transaction value
        and the week of year.
        - Vl_per_dayWk: ration between the transaction value
        and the day of week.
        - Vl_per_dayYr: ration between the transaction value
        and the day of year.

    Args:
        data (spark data frame): input spark data frame.
    """
    data = data.withColumn('Vl_per_weekYr',
                           (data['Value'] / data['WeekOfYear']))
    data = data.withColumn('Vl_per_dayWk',
                           (data['Value'] / data['DayOfWeek']))
    data = data.withColumn('Vl_per_dayYr',
                           (data['Value'] / data['DayOfYear']))
    return data


def create_features_for_transaction_value_and_category(data):
    """Create new features relating the transaction value for
    each product category.

    Args:
        data (spark data frame): input spark data frame.
    """
    for item in cfg.LIST_OF_TRANSACTION_CATEGORIES:
        data = create_feature_average_value_for_category(data, item)
        data = create_feature_ratio_between_value_and_category(data, item)
    return data


def create_feature_average_value_for_category(data, item):
    """Create a new feature with average transaction value for
    all aggregated by item.

    Args:
        data (spark data frame): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.

    Returns:
        data (spark data frame): output spark data frame with
        the new feature created.
    """
    type_column = 'avg'
    column_name = '{0}_vl_{1}'.format(type_column, item)
    aux = data.select([item, 'Value']).\
        groupBy(item).\
        mean()
    aux = aux.select(col(item),
                     col(type_column +
                     '(Value)').alias(column_name))
    data = data.join(aux, on=item)
    return data


def create_feature_ratio_between_value_and_category(data, item):
    """Create a new feature with the ratio between the transaction
    value and the product category.

    Args:
        data (spark data frame): input spark data frame.
        item: type of attribute used to aggregate the data and
        compute the average.
    """
    column_name = 'avg_vl_{0}'.format(item)
    ratio_column_name = 'rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col('Value') - col(column_name)) /
                           col(column_name))
    return data


def set_df_from_data_train(**kwargs):
    logging.info('Splitting data')
    # converting to pandas
    data = cfg.x_data_temp
    # get back the data
    cfg.x_data_temp = data[cfg.ALL_FEATURES]
    if 'test' not in kwargs:
        cfg.y_train = data[cfg.LABEL]
    # set frauds : outliers
    cfg.x_outliers = data[data[cfg.LABEL].isin([1])]
    # set df to train
    cfg.x_train_numerical = data[cfg.NUMERICAL_FEATURES]
    # set df fraud to train
    cfg.x_outliers_numerical = cfg.x_outliers[cfg.NUMERICAL_FEATURES]


def create_feature_outlier_intensity():
    """Each model (Isolation Forest, LSCP, and KNN) classify the instance
    into outlier or not. This function create a new feature with the number
    of models that classify the instance as outlier. For example:
    - if only KNN model classify the instance as outlier, this value will be 1
    - if LSCP and KNN models classify the instance as outlier, so this value
    will be 2
    - if all models classify the instance as outlier, so this value will be 3
    - if none model is outlier, so this value will be 0
    """
    logging.info('Creating outliers detections features')
    cfg.x_data_temp[cfg.COUNT_COLUMN_NAME] = \
        (cfg.x_data_temp.IsolationForest +
         cfg.x_data_temp.LSCP +
         cfg.x_data_temp.KNN)
    add_outlier_to_features_list()


def add_outlier_to_features_list():
    """The variables CATEGORICAL_FEATURES and ALL_FEATURES
    keep the relation of categorical features, so this
    function update these lists.
    """
    new_features_list = [cfg.IF_COLUMN_NAME,
                         cfg.LSCP_COLUMN_NAME,
                         cfg.KNN_COLUMN_NAME,
                         cfg.COUNT_COLUMN_NAME]
    cfg.CATEGORICAL_FEATURES += new_features_list
    cfg.ALL_FEATURES += new_features_list
    cfg.categorical_features_dims = \
        [cfg.x_data_temp.columns.get_loc(i)
         for i in cfg.CATEGORICAL_FEATURES[:]]


def split_train_val(**kwargs):
    """Split arrays or matrices into random train and test subsets.
    Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    logging.info('Splitting data Train/Validation')
    cfg.x_train, cfg.x_validation, cfg.y_train, cfg.y_validation = \
        sk_ml.train_test_split(cfg.x_train[cfg.ALL_FEATURES],
                               cfg.y_train,
                               test_size=cfg.TEST_SPLIT_SIZE,
                               random_state=cfg.RANDOM_NUMBER)
    save_validation_data_in_csv_format(**kwargs)


def save_validation_data_in_csv_format(**kwargs):
    """Saving the validation data in CSV format.
    """
    logging.info('Exporting data Validation')
    cfg.x_validation.to_csv(kwargs['output_valid_x_file'],
                       index=None, header=True)
    cfg.y_validation.to_csv(kwargs['output_valid_y_file'],
                       index=None, header=True)


def balance_oversampling(**kwargs):
    """Use the SMOTENC to create oversampling for input data set.
    Save this data in disk using the CSV format.
    Plot a distribution for class after balancing the data.
    """
    logging.info('Balancing the train data')
    x_train, y_train = models.smotenc_over_sampler()
    cfg.x_train_balanced = pd.DataFrame(x_train, columns=cfg.ALL_FEATURES)
    cfg.y_train_balanced = pd.DataFrame(y_train, columns=[cfg.LABEL])
    save_training_data_in_csv_format(**kwargs)
    vis.plot_target_distribution()


def save_training_data_in_csv_format(**kwargs):
    """Saving the validation data in CSV format.
    """
    logging.info(save_training_data_in_csv_format.__name__)
    cfg.x_train_balanced.to_csv(kwargs['output_balanced_train_x_file'],
                                index=False)
    cfg.y_train_balanced.to_csv(kwargs['output_balanced_train_y_file'],
                                index=False)


def extract_data_balanced(**kwargs):
    logging.info(extract_data_balanced.__name__)
    cfg.x_train_balanced = pd.read_csv(kwargs['output_balanced_train_x_file'])
    cfg.y_train_balanced = pd.read_csv(kwargs['output_balanced_train_y_file'])


def train_model():
    logging.info(train_model.__name__)
    models.train_cat_boost()
    vis.plot_feature_importance()


def evaluate_model(mode):
    logging.info(evaluate_model.__name__)
    models.predict_cat_boost(mode)
    if mode == 'VALID':
        export_cat_boost_validate()


def export_cat_boost_validate():
    logging.info(export_cat_boost_validate.__name__)
    precision, recall, f_score, _ = \
        score(cfg.x_to_predict_catboost['FraudResult'],
              cfg.x_to_predict_catboost['CatBoost'])

    out_catb_file = open('../data/catBoost_model_result.txt', 'w')
    out_catb_file.write('LABELS\t\tFraudResult\t\t\t\t | \tCatBoost\n')
    out_catb_file.write('------------------------------------------\n')
    out_catb_file.write('precision: \t{}\t\t | \t{}\n'.
                        format(precision[0], precision[1]))
    out_catb_file.write('recall: \t\t{}\t\t | \t{}\n'.
                        format(recall[0], recall[1]))
    out_catb_file.write('f-score: \t\t{}\t\t | \t{}\n'.
                        format(f_score[0], f_score[1]))
    out_catb_file.write('------------------------------------------\n')
    out_catb_file.write('CAT-BOOST CONFIGURATION--------------------\n')
    out_catb_file.write('depth: {} - LR {} - L2: {}\n'.
                        format(cfg.DEPTH_CATBOOST,
                               cfg.LEARNING_RATE_CATBOOST,
                               cfg.L2_CATBOOST))
    out_catb_file.close()


def export_data_valid_result(**kwargs):
    logging.info(export_data_valid_result.__name__)
    print('{}'.format(kwargs['output_valid_result_file']))
    cfg.x_to_predict_catboost.to_csv(kwargs['output_valid_result_file'],
                                     index=None, header=True)


def export_data_test_result(**kwargs):
    logging.info(export_data_test_result.__name__)
    save_predictions_xente(kwargs['output_test_result_file'],
                           cfg.x_to_predict_catboost['TransactionId'],
                           cfg.x_to_predict_catboost['CatBoost'])


def is_missing_file_test(**kwargs):
    logging.info('Finding Data Balanced to Valid')
    ans = os.path.exists(kwargs['input_test_file'])
    return ans


def there_is_duplicate_lines(data):
    ans = data.count() != data.distinct().count()
    print('There is duplicated lines? {0}'.format(ans))


def get_specific_statistical_info(data, stat):
    """
    :param data: PySpark data frame.
    :param stat: could be "avg", "max", or "min".
    :return: The respective statistics over input data.
    """
    if stat == "avg":
        data = data.mean()
    elif stat == "max":
        data = data.max()
    elif stat == "min":
        data = data.min()
    return data


def save_predictions_xente(file_name, transactions_list, test_pred):
    """
    :param file_name:
    :param transactions_list:
    :param test_pred:
    :return:
    """
    file = open(file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for trans_id, value in zip(transactions_list, test_pred):
        file.write('{0},{1}\n'.format(trans_id, int(value)))
    file.close()


def get_categorical_features():
    return cfg.CATEGORICAL_FEATURES
