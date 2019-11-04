import parameters as param
import preprocessing as preprocess
import visualization as vis
import statistics as stats
import os
import sys

sys.path.insert(0, 'models/')
import isolation_forest

# This param when defined as True will execute
# the complete code, so slowly processing time
# because is require to execute all checks and
# print all not essential functions.
# When defined as False, a fast processing is
# applied with all core functionalities working
# well.
full_execution = False
verbose_mode = True

os.environ["JAVA_HOME"] = param.get_java_home()
os.environ["SPARK_HOME"] = param.get_spark_home()

import findspark

findspark.init()
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('xente').getOrCreate()
spark

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyspark.sql.functions as F
import shap
import catboost
from catboost import Pool, CatBoostClassifier, cv
from sklearn.ensemble import IsolationForest
from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import mean, udf, array, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score)

all_features = ['ProductId', 'ProductCategory', 'ChannelId', 'Value', 'PricingStrategy', 'Operation', 'PositiveAmount',
                'avg_ps_ChannelId', 'rt_avg_ps_ChannelId', 'avg_ps_ProductCategory', 'rt_avg_ps_ProductCategory',
                'avg_ps_ProductId', 'rt_avg_ps_ProductId']
feature_cols = ['PositiveAmount', 'Operation', 'Value', 'PricingStrategy']
columns_to_remove = ['CurrencyCode', 'CountryCode', 'BatchId', 'AccountId', 'SubscriptionId',
                     'CustomerId', 'TransactionStartTime', 'Amount']
categorical_features = ['ProductId', 'ProductCategory', 'ChannelId']
numerical_features_augmented = ['Value', 'PricingStrategy', 'Operation', 'PositiveAmount', 'avg_ps_ChannelId',
                                'rt_avg_ps_ChannelId', 'avg_ps_ProductCategory', 'rt_avg_ps_ProductCategory',
                                'avg_ps_ProductId', 'rt_avg_ps_ProductId']
label = ['FraudResult']
outliers_label = 'FraudResult==1'


def read_spark_data_frame(file_name):
    data = pd.read_csv(file_name)
    spark_data = spark.createDataFrame(data)
    return spark_data


# Read Fraud Detection Challenge data
train_data = read_spark_data_frame(param.get_file_name('training_data'))

# Create new features and remove the non used features
train_data = preprocess.get_features_augmentation(train_data)
train_data = train_data.drop(*columns_to_remove)

# Print Description details about the data set
if full_execution:
    # Checking if there are missing data or duplicate line?
    preprocess.there_is_missing_data(train_data)
    preprocess.there_is_duplicate_lines(train_data)
    # Plot transactions proportion comparing fraudulent with genuine transactions
    vis.plot_transactions_proportions(train_data)
    # Print a full description over the data
    stats.print_description(train_data, feature_cols)
    # Plot histogram distribution for all features
    # True is used for genuine data
    # False is used for fraudulent data
    vis.plot_hist(train_data, feature_cols, True)
    vis.plot_hist(train_data, feature_cols, False)
    # Plot correlation matrix for fraudulent and genuine data
    vis.plot_heatmap(train_data, numerical_features_augmented, True)
    vis.plot_heatmap(train_data, numerical_features_augmented, False)

    model = isolation_forest.IsolationForest(train_data, numerical_features_augmented, label, outliers_label)
    max_sample_list = [100, 200, 800, 1600, 3200, 6400]
    contamination_list = [0.003, 0.01, 0.05, 0.1]
    _, _, _, _, score = model.fit_grid_search(max_sample_list, contamination_list, metric='f1score')
    vis.plot_performance_comparison(score)

# TO DO: Oversampling models (RandomOverSampler, SMOTENC, ADASYN, etc)

# TO DO: Feature importance using SHAP

# TO DO: Catboost

print('Finish with success')
