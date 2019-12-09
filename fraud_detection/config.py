from pyspark.sql import SparkSession
import findspark
findspark.init()

spark = SparkSession.builder.appName('Xente').getOrCreate()

ITEMS_LIST = ['ProductId', 'ProviderId']
COLUMN_VALUE = 'Value'
LABEL = 'FraudResult'
ALL_FEATURES = ['ProviderId', 'ProductId', 'TransactionId',
                'BatchId', 'ProductCategory', 'ChannelId',
                'PricingStrategy', 'Value', 'Operation', 'Hour',
                'DayOfWeek', 'WeekOfYear',
                'Vl_per_weekYr', 'Vl_per_dayWk',
                'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']
ALL_FEATURES_TEST = ['ProviderId', 'ProductId', 'TransactionId',
                     'BatchId', 'ProductCategory', 'ChannelId',
                     'PricingStrategy', 'Value', 'Operation', 'Hour',
                     'DayOfWeek', 'WeekOfYear',
                     'Vl_per_weekYr', 'Vl_per_dayWk',
                     'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']
CATEGORICAL_FEATURES = ['ProviderId', 'ProductId', 'TransactionId',
                        'BatchId', 'ProductCategory', 'ChannelId',
                        'PricingStrategy']
CATEGORICAL_FEATURES_TEST = ['ProviderId', 'ProductId', 'TransactionId',
                             'BatchId', 'ProductCategory', 'ChannelId',
                             'PricingStrategy']
NUMERICAL_FEATURES = ['Value', 'Operation', 'Hour',
                      'DayOfWeek', 'WeekOfYear',
                      'Vl_per_weekYr', 'Vl_per_dayWk',
                      'rt_avg_vl_ProductId', 'rt_avg_vl_ProviderId']

categorical_features_dims = 0
all_features_dims = 0

data_train = []
data_test = []
x_data_temp = []
x_train = []
y_train = []
x_train_balanced = []
y_train_balanced = []
x_valid = []
y_valid = []
x_outliers = []
x_train_numerical = []
x_outliers_numerical = []
predictions = []
x_to_predict_catboost = []

TEST_SPLIT_SIZE = 0.3
LEARNING_RATE_LIST = [0.0001, 0.001, 0.005, 0.01, 0.1]
LEAF_REG_LIST = [4, 5, 6, 7, 8]
DEPTH_LIST = [4, 5, 6, 7, 8]

DEPTH_CATBOOST = 5
LEARNING_RATE_CATBOOST = 0.1
L2_CATBOOST = 2
EVAL_METRIC = 'F1'
TYPE_DEVICE_CATBOOST = 'GPU'
RANDOM_NUMBER = 42
model_catboost_file = '../data/catBoost_model'

percent_contamination = 0.00201752001839811
NUM_NEIGHBORS = 3
NUM_ESTIMATORS = 3
NUM_CLUSTERS = 2
N_JOBS = 12

model_cat_boost = []
model_if = []
model_feat_bag = []
model_lof = []
model_cblof = []
model_lscp = []
model_knn = []
model_smotenc = []

IF_COLUMN_NAME = 'IsolationForest'
LSCP_COLUMN_NAME = 'LSCP'
KNN_COLUMN_NAME = 'KNN'
COUNT_COLUMN_NAME = 'CountDetection'
