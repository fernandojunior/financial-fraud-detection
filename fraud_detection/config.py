from os import path
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Xente').getOrCreate()

ITEMS_LIST = ['ProductId', 'ProviderId']
COLUMN_NAME = "VALUE"
LABEL = 'FraudResult'

ITEMS_TO_BE_REMOVED_LIST = ['CurrencyCode', 'CountryCode', 'AccountId',
                            'SubscriptionId', 'CustomerId', 'TransactionStartTime',
                            'Amount', 'DayOfYear', 'Avg_vl_ProductId', 'Avg_vl_ProviderId']

ALL_FEATURES = ['ProviderId', 'ProductId', 'TransactionId', 'BatchId',
                'ProductCategory', 'ChannelId', 'PricingStrategy', 'ValueStrategy',
                'Value', 'Operation', 'Hour', 'DayOfWeek', 'WeekOfYear',
                'Vl_per_weekYr', 'Vl_per_dayWk', 'Vl_per_dayYr',
                'Rt_avg_vl_ProductId', 'Rt_avg_vl_ProviderId']

CATEGORICAL_FEATURES = ['ProviderId', 'ProductId', 'TransactionId', 'BatchId',
                        'ProductCategory', 'ChannelId', 'PricingStrategy']

NUMERICAL_FEATURES = ['ValueStrategy', 'Value', 'Operation', 'Hour', 'DayOfWeek', 'WeekOfYear',
                      'Vl_per_weekYr', 'Vl_per_dayWk', 'Vl_per_dayYr',
                      'Rt_avg_vl_ProductId', 'Rt_avg_vl_ProviderId']

LEARNING_RATE_LIST = [0.0001, 0.001, 0.005, 0.01, 0.1]
DEPTH_LIST = [4, 5, 6, 7, 8]
LEAF_REG = [4, 5, 6, 7, 8]

IF_COLUMN_NAME = 'IsolationForest'
LSCP_COLUMN_NAME = 'LSCP'
KNN_COLUMN_NAME = 'KNN'
COUNT_COLUMN_NAME = 'CountDetection'
EVAL_METRIC = 'F1'

NUM_NEIGHBORS = 2
RANDOM_NUMBER = 42
N_JOBS = 10

contamination_level = 0
categorical_features_dims = 0
numerical_features_dims = 0

data_train = []
x_train = []
y_train = []
x_outliers = []
x_train_numerical = []
x_outliers_numerical = []
x_train_balanced = []
y_train_balanced = []

data_test = []

model_cat_boost = []
model_isolation_forest = []
model_lscp = []
model_knn = []