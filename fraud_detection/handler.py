import logging

import pandas as pd
from pyspark.sql.functions import when, col, hour, dayofweek, month, weekofyear, dayofyear, abs

import config as cfg
import models
import visualization as vis

logging.basicConfig(filename='log_file.log',
                    level=logging.INFO,
                    filemode='w',
                    format='%(asctime)s :: %(filename)s :: %(funcName)s\n\t%(message)s')


def outside_log(s1, s2):
    logging.info(s1 + '.py :: ' + s2)


def extract_data_train(**kwargs):
    logging.info('Extracting Data')
    cfg.data_train = read_data(kwargs['input_train_file'])


def extract_data_test(**kwargs):
    logging.info('Extracting Data')
    cfg.data_test = read_data(kwargs['input_test_file'])


def read_data(file_name):
    """
    Return data in spark dataframe format.
    :param file_name: Input file name for dataset, could be a url or
    a path to a local file.
    :return: spark dataframe.
    """
    data = cfg.spark.read.csv(file_name, header=True, inferSchema=True)
    return data


def handle_data_train(**kwargs):
    logging.info('Handling Data Train')
    # set contamination value
    set_contamination()
    # create new features
    create_features(cfg.data_train)
    # plot heat-map of features
    vis.plot_heatmap('INIT')
    # split data in train and validation
    set_df_from_data_train(**kwargs)
    # outliers
    models.train_isolation_forest()
    models.predict_isolation_forest()
    models.train_lscp()
    models.predict_lscp()
    models.train_knn()
    models.predict_knn()
    create_outlier_features()
    # get back data to x_train
    cfg.x_train = cfg.x_data_temp
    vis.plot_heatmap('OUTLIER')


def handle_data_test():
    logging.info('Handling Data Test')
    # create new features
    create_features(cfg.data_test)
    cfg.ALL_FEATURES = cfg.ALL_FEATURES_TEST
    cfg.CATEGORICAL_FEATURES = cfg.CATEGORICAL_FEATURES
    models.predict_isolation_forest()
    models.predict_lscp()
    models.predict_knn()
    create_outlier_features()
    cfg.x_to_predict_catboost = cfg.x_data_temp


def set_contamination():
    logging.info('Get contamination into the data')
    cfg.percent_contamination = (cfg.data_train.filter('FraudResult==1').count()) / (cfg.data_train.count())


def create_features(df):
    logging.info('Creating pre-defined features')
    cfg.x_data_temp = df
    cfg.x_data_temp = generate_new_features(cfg.x_data_temp)
    cfg.x_data_temp = cfg.x_data_temp.toPandas()


def generate_new_features(data):
    data = get_type_operation(data)
    data = get_value_strategy(data)
    data = get_timestamp(data)
    data = get_value_per_period(data)
    data = get_features_per_value(data)
    return data


def get_type_operation(data):
    data = data.withColumn("Operation", when(data.Amount > 0, 1).when(data.Amount < 0, -1).otherwise(0))
    return data


def get_value_strategy(data):
    avg_value = data.agg({cfg.COLUMN_VALUE: 'avg'}).collect()[0][0]
    data = data.withColumn('ValueStrategy', when(col(cfg.COLUMN_VALUE) > avg_value * 100, 3)
                                            .when(col(cfg.COLUMN_VALUE) > avg_value * 10, 2)
                                            .when(col(cfg.COLUMN_VALUE) > avg_value * 2, 1)
                           .otherwise(0))
    return data


def get_timestamp(data):
    data = data.withColumn('Hour', hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', weekofyear(data['TransactionStartTime']))
    return data


def get_value_per_period(data):
    data = data.withColumn('Vl_per_weekYr', (data['Value'] / data['WeekOfYear']))
    data = data.withColumn('Vl_per_dayWk', (data['Value'] / data['DayOfWeek']))
    data = data.withColumn('Vl_per_dayYr', (data['Value'] / data['DayOfYear']))
    return data


def get_features_per_value(data):
    for item in cfg.ITEMS_LIST:
        data = get_value_average(data, item)
        data = get_value_ratio(data, item)
    return data


def get_value_average(data, item):
    type_column = 'avg'
    column_name = '{0}_vl_{1}'.format(type_column, item)
    aux = data.select([item, 'Value']).groupBy(item).mean()
    aux = aux.select(col(item), col(type_column + '(Value)').alias(column_name))
    data = data.join(aux, on=item)
    return data


def get_value_ratio(data, item):
    column_name = 'avg_vl_{0}'.format(item)
    ratio_column_name = 'rt_avg_vl_{0}'.format(item)
    data = data.withColumn(ratio_column_name,
                           (col('Value') - col(column_name)) / col(column_name))
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


def create_outlier_features():
    logging.info('Creating outliers detections features')
    cfg.x_data_temp[cfg.COUNT_COLUMN_NAME] = (cfg.x_data_temp.IsolationForest +
                                              cfg.x_data_temp.LSCP +
                                              cfg.x_data_temp.KNN)
    add_outlier_features_to_list()


def add_outlier_features_to_list():
    new_features_list = [cfg.IF_COLUMN_NAME,
                         cfg.LSCP_COLUMN_NAME,
                         cfg.KNN_COLUMN_NAME,
                         cfg.COUNT_COLUMN_NAME]
    cfg.CATEGORICAL_FEATURES += new_features_list
    cfg.ALL_FEATURES += new_features_list
    cfg.categorical_features_dims = [cfg.x_data_temp.columns.get_loc(i) for i in cfg.CATEGORICAL_FEATURES[:]]


def split_train_val(**kwargs):
    logging.info('Splitting data Train/Validation')
    import sklearn.model_selection as sk_ml
    cfg.x_train, cfg.x_valid, cfg.y_train, cfg.y_valid = sk_ml.train_test_split(cfg.x_train[cfg.ALL_FEATURES],
                                                                                cfg.y_train,
                                                                                test_size=cfg.TEST_SPLIT_SIZE,
                                                                                random_state=cfg.RANDOM_NUMBER)
    export_csv_to_validation(**kwargs)


def export_csv_to_validation(**kwargs):
    logging.info('Exporting data Validation')
    cfg.x_valid.to_csv(kwargs['output_valid_x_file'], index=None, header=True)
    cfg.y_valid.to_csv(kwargs['output_valid_y_file'], index=None, header=True)


def balance_oversampling(**kwargs):
    logging.info('Balancing the train data')
    from models import smotenc_over_sampler
    x, y = smotenc_over_sampler()
    cfg.x_train_balanced = pd.DataFrame(x, columns=cfg.ALL_FEATURES)
    cfg.y_train_balanced = pd.DataFrame(y, columns=[cfg.LABEL])
    export_data_balanced(**kwargs)
    vis.plot_target_distribution()


def export_data_balanced(**kwargs):
    logging.info(export_data_balanced.__name__)
    cfg.x_train_balanced.to_csv(kwargs['output_balanced_train_x_file'], index=False)
    cfg.y_train_balanced.to_csv(kwargs['output_balanced_train_y_file'], index=False)


def extract_data_balanced(**kwargs):
    logging.info(extract_data_balanced.__name__)
    cfg.x_train_balanced = pd.read_csv(kwargs['output_balanced_train_x_file'])
    cfg.y_train_balanced = pd.read_csv(kwargs['output_balanced_train_y_file'])


def train_model():
    logging.info(train_model.__name__)
    models.train_cat_boost()
    vis.plot_feature_importance()


def extract_data_validation(**kwargs):
    logging.info(extract_data_balanced.__name__)
    cfg.x_valid = pd.read_csv(kwargs['output_valid_x_file'])
    cfg.y_valid = pd.read_csv(kwargs['output_valid_y_file'])
    cfg.x_to_predict_catboost = cfg.x_valid


def evaluate_model(mode):
    logging.info(evaluate_model.__name__)
    models.predict_cat_boost(mode)
    if mode is 'VALID':
        export_cat_boost_validate()


def export_cat_boost_validate():
    logging.info(export_cat_boost_validate.__name__)
    from sklearn.metrics import precision_recall_fscore_support as score
    precision, recall, f_score, support = score(cfg.x_to_predict_catboost['FraudResult'],
                                                cfg.x_to_predict_catboost['CatBoost'])

    out_catb_file = open('../data/catBoost_model_result.txt', 'w')
    out_catb_file.write('LABELS\t\tFraudResult\t\t\t\t | \tCatBoost\n')
    out_catb_file.write('------------------------------------------\n')
    out_catb_file.write('precision: \t{}\t\t | \t{}\n'.format(precision[0], precision[1]))
    out_catb_file.write('recall: \t\t{}\t\t | \t{}\n'.format(recall[0], recall[1]))
    out_catb_file.write('f-score: \t\t{}\t\t | \t{}\n'.format(f_score[0], f_score[1]))
    out_catb_file.write('------------------------------------------\n')
    out_catb_file.write('CAT-BOOST CONFIGURATION--------------------\n')
    out_catb_file.write('depth: {} - LR {} - L2: {}\n'.format(cfg.DEPTH_CATBOOST,
                                                              cfg.LEARNING_RATE_CATBOOST,
                                                              cfg.L2_CATBOOST))
    out_catb_file.close()


def export_data_valid_result(**kwargs):
    logging.info(export_data_valid_result.__name__)
    print('{}'.format(kwargs['output_valid_result_file']))
    cfg.x_to_predict_catboost.to_csv(kwargs['output_valid_result_file'], index=None, header=True)


def export_data_test_result(**kwargs):
    logging.info(export_data_test_result.__name__)
    save_predictions_xente(kwargs['output_test_result_file'],
                           cfg.x_to_predict_catboost['TransactionId'],
                           cfg.x_to_predict_catboost['CatBoost'])


def is_missing_file_test(**kwargs):
    logging.info('Finding Data Balanced to Valid')
    import os.path
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


def get_features_augmentation(data):
    """
    This function was create to make features augmentation in the data to improve
    the performance model.
    :param data:
    :return:
        - Operation: The Amount value corresponds to transaction value,
        if positive value indicates a debit transaction, while a negative
        value indicates a credit transaction. So, we map this OPERATION
        to -1 and 1, in which -1 was used for credit and 1 for debit.
        - PositiveAmount: There is some difference between the Amount and
        Value column, Value is suppose to be the absolute value of Amount.
        So, we calculate the PositiveAmount, which corresponds to the
        absolute value of Amount, after that we discard this attribute.
        - We had three categorical features: ChannelId, ProductCategory, and
        ProductId. We decide create new features based on this, they are
        repeated process for the three categories:
            - Avg_ps: average of positive amount for a same category product,
            e.g., there is different channels, each channels has a mean value
            for all transactions of this category, so a new column was added
            to data frame including this value, indicating the mean value for
            this category product.
            - Rt_avg_ps: this feature calculate the ratio between the transaction
             value instance and the global mean. How
    """
    data = data.withColumn("Operation", when(data.Amount > 0, 1).when(data.Amount < 0, -1).otherwise(0))
    data = data.withColumn('PositiveAmount', abs(data['Amount']))

    data = data.withColumn('Hour', hour(data['TransactionStartTime']))
    data = data.withColumn('DayOfWeek', dayofweek(data['TransactionStartTime']))
    data = data.withColumn('DayOfYear', dayofyear(data['TransactionStartTime']))
    data = data.withColumn('WeekOfYear', weekofyear(data['TransactionStartTime']))
    data = data.withColumn('Month', month(data['TransactionStartTime']))

    data = data.withColumn('Ps_per_dayWk', (data['PositiveAmount'] / data['DayOfWeek']))
    data = data.withColumn('Ps_per_dayYr', (data['PositiveAmount'] / data['DayOfYear']))
    data = data.withColumn('Op_x_value', (data['Operation'] * data['Value']))

    items_list = ['ChannelId', 'ProductCategory', 'ProductId']
    for item in items_list:
        for statistical_type in ["avg", "min", "max"]:
            column_name = "{0}_ps_{1}".format(statistical_type, item)
            aux = get_specific_statistical_info(
                data.select([item, 'PositiveAmount']).groupBy(item), statistical_type)
            aux = aux.select(col(item), col("{0}(PositiveAmount)".format(statistical_type)).alias(column_name))
            data = data.join(aux, on=item)
            if statistical_type == "avg":
                ratio_column_name = 'rt_avg_ps_{0}'.format(item)
                data = data.withColumn(ratio_column_name, (col('PositiveAmount') - col(column_name)) / col(column_name))
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


def is_missing_data():
    logging.info('Finding Data')
    ans = (cfg.data_train.count() != cfg.data_train.na.drop(how='any').count())
    return ans


def is_missing_file_train(**kwargs):
    logging.info('Finding Data Balanced to Train')
    import os.path
    ans = (os.path.exists(kwargs['output_x_file_name']) and
           os.path.exists(kwargs['output_y_file_name']))
    return ans


def is_missing_file_validation(**kwargs):
    logging.info('Finding Data Balanced to Valid')
    import os.path
    ans = os.path.exists(kwargs['output_valid_result_file'])
    return ans


def get_categorical_features():
    return cfg.CATEGORICAL_FEATURES
