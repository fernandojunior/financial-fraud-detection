from os import path
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Xente').getOrCreate()

ITEMS_LIST = ['ProductId', 'ProviderId']
COLUMN_VALUE = 'VALUE'
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

data_valid = []

LEARNING_RATE_LIST = [0.0001, 0.001, 0.005, 0.01, 0.1]
LEAF_REG_LIST = [4, 5, 6, 7, 8]
DEPTH_LIST = [4, 5, 6, 7, 8]

DEPTH_CATBOOST = 5
LEARNING_RATE_CATBOOST = 0.1
L2_CATBOOST = 1
EVAL_METRIC = ['F1','AUC']
TYPE_DEVICE_CATBOOST = 'GPU'
NUM_DEVICE_CATBOOST = '0'
RANDOM_NUMBER = 42
model_catboost_saved = '../data/catBoost_model.dump'

model_cat_boost = []
from catboost import CatBoostClassifier
catboost_classifier = CatBoostClassifier(depth=DEPTH_CATBOOST,
    learning_rate=LEARNING_RATE_CATBOOST,
    l2_leaf_reg=L2_CATBOOST,
    eval_metric=EVAL_METRIC,
    use_best_model=True,
    task_type=TYPE_DEVICE_CATBOOST,
    devices=NUM_DEVICE_CATBOOST,
    random_seed=RANDOM_NUMBER)

CONTAMINATION = 0.5
NUM_NEIGHBORS = 2
NUM_ESTIMATORS = 2
N_JOBS = 12

IF_COLUMN_NAME = 'IsolationForest'
set_isolation_forest = []
from sklearn.ensemble import IsolationForest
if_outlier = IsolationForest(behaviour='new',
    random_state=RANDOM_NUMBER,
    contamination=CONTAMINATION,
    n_jobs=N_JOBS)

model_feature_bagging = []       
from pyod.models.feature_bagging import FeatureBagging
featBag_outlier = FeatureBagging(contamination=CONTAMINATION, 
    combination='max', 
    n_estimators=NUM_ESTIMATORS, 
    random_state=RANDOM_NUMBER, 
    n_jobs=N_JOBS)

model_lof = []
from pyod.models.lof import LOF
lof_outlier = LOF(contamination=CONTAMINATION, 
    n_neighbors=NUM_NEIGHBORS,
    n_jobs=N_JOBS)

LSCP_COLUMN_NAME = 'LSCP'
model_lscp = []
from pyod.models.lscp import LSCP
lscp_outlier = LSCP([featBag_outlier, lof_outlier],
    contamination=CONTAMINATION,
    random_state=RANDOM_NUMBER)

KNN_COLUMN_NAME = 'KNN'
model_knn = []
from pyod.models.knn import KNN
knn_outlier = KNN(contamination=CONTAMINATION,
    n_neighbors=NUM_NEIGHBORS,
    method='mean',
    n_jobs=N_JOBS)

from imblearn.over_sampling import SMOTENC
smotenc_oversampler = SMOTENC(categorical_features=categorical_features_dims,
    random_state=RANDOM_NUMBER, 
    n_jobs=N_JOBS)

COUNT_COLUMN_NAME = 'CountDetection'