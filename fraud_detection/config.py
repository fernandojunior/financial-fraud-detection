from os import path
import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('Xente').getOrCreate()

ITEMS_LIST = ['ProductId', 'ProviderId']
COLUMN_VALUE = 'Value'
LABEL = 'FraudResult'
ALL_FEATURES = ['ProviderId','ProductId','TransactionId',
                'BatchId','ProductCategory','ChannelId',
                'PricingStrategy','Value','Operation','Hour',
                'DayOfWeek','WeekOfYear',
                'Vl_per_weekYr','Vl_per_dayWk',
                'rt_avg_vl_ProductId','rt_avg_vl_ProviderId']
CATEGORICAL_FEATURES = ['ProviderId','ProductId','TransactionId',
                        'BatchId','ProductCategory','ChannelId',
                        'PricingStrategy']
NUMERICAL_FEATURES = ['Value','Operation','Hour',
                      'DayOfWeek','WeekOfYear',
                      'Vl_per_weekYr','Vl_per_dayWk',
                      'rt_avg_vl_ProductId','rt_avg_vl_ProviderId']

categorical_features_dims = 0
all_features_dims = 0

data_train = []
data_test = []
x_train = []
y_train = []
x_valid = []
y_valid = []
x_outliers = []
x_train_numerical = []
x_outliers_numerical = []


predictions = []

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

model_cat_boost = []
def set_catboost_model():
    from catboost import CatBoostClassifier
    catboost_classifier = CatBoostClassifier(depth=DEPTH_CATBOOST,
        learning_rate=LEARNING_RATE_CATBOOST,
        l2_leaf_reg=L2_CATBOOST,
        eval_metric=EVAL_METRIC,
        task_type=TYPE_DEVICE_CATBOOST,
        random_seed=RANDOM_NUMBER)
    return catboost_classifier

def get_catFeat():
    return CATEGORICAL_FEATURES

percent_contamination = 0.00201752001839811
NUM_NEIGHBORS = 3
NUM_ESTIMATORS = 3
NUM_CLUSTERS = 2
N_JOBS = 12

IF_COLUMN_NAME = 'IsolationForest'
set_isolation_forest = []
from sklearn.ensemble import IsolationForest
if_outlier = IsolationForest(behaviour='new',
    random_state=RANDOM_NUMBER,
    contamination=percent_contamination,
    n_jobs=N_JOBS)

model_feature_bagging = []       
from pyod.models.feature_bagging import FeatureBagging
featBag_outlier = FeatureBagging(contamination=percent_contamination, 
    combination='max', 
    n_estimators=NUM_ESTIMATORS, 
    random_state=RANDOM_NUMBER, 
    n_jobs=N_JOBS)

model_lof = []
from pyod.models.lof import LOF
lof_outlier = LOF(contamination=percent_contamination, 
    n_neighbors=NUM_NEIGHBORS,
    n_jobs=N_JOBS)

model_cblof = []
from pyod.models.cblof import CBLOF
cblof_outlier = CBLOF(contamination=percent_contamination, 
    n_clusters=NUM_CLUSTERS,
    random_state=RANDOM_NUMBER,
    n_jobs=N_JOBS)

LSCP_COLUMN_NAME = 'LSCP'
model_lscp = []
from pyod.models.lscp import LSCP
lscp_outlier = LSCP(detector_list=[featBag_outlier, 
                                   lof_outlier, 
                                   cblof_outlier],
                    contamination=percent_contamination,
                    random_state=RANDOM_NUMBER)

KNN_COLUMN_NAME = 'KNN'
model_knn = []
from pyod.models.knn import KNN
knn_outlier = KNN(contamination=percent_contamination,
    n_neighbors=NUM_NEIGHBORS,
    method='mean',
    n_jobs=N_JOBS)

COUNT_COLUMN_NAME = 'CountDetection'

# A chamada do modelo deve ser via funcao
# devido a atualizacao da variavel #categorical_features_dims
# durante execucao do codigo
def set_smotenc_model():
    from imblearn.over_sampling import SMOTENC
    smotenc_model = SMOTENC(categorical_features=categorical_features_dims,
        random_state=RANDOM_NUMBER, 
        n_jobs=N_JOBS)
    return smotenc_model
