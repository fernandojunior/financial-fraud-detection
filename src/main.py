import parameters as param
import preprocessing as preprocess
import os

# This param when defined as True will execute
# the complete code, so slowly processing time
# because is require to execute all checks and
# print all not essential functions.
# When defined as False, a fast processing is
# applied with all core functionalities working
# well.
full_execution = True

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

from pyspark.sql.types import IntegerType, DoubleType
from pyspark.sql.functions import mean, udf, array, col
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.linalg import Vectors
from imblearn.over_sampling import RandomOverSampler, SMOTE, SMOTENC, ADASYN
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split, learning_curve, ShuffleSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score)


def read_spark_data_frame(file_name):
    data = pd.read_csv(file_name)
    spark_data = spark.createDataFrame(data)
    return spark_data


# Read Fraud Detection Challenge data
train_data = read_spark_data_frame(param.get_file_name('training_data'))


# Print Description details about the data set
if full_execution:
    preprocess.there_is_missing_data(train_data)
    preprocess.there_is_duplicate_lines(train_data)


train_data = preprocess.get_features_augmentation(train_data)

# Remove non used features
columns_to_remove = ['CurrencyCode', 'CountryCode', 'BatchId', 'AccountId', 'SubscriptionId', 'CustomerId', 'TransactionStartTime', 'Amount']
train_data = train_data.drop(*columns_to_remove)


print(train_data.show())
print('Finish with success')
