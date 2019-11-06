import parameters as param
import os
os.environ["JAVA_HOME"] = param.get_java_home()
os.environ["SPARK_HOME"] = param.get_spark_home()

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('xente').getOrCreate()
import pandas as pd


def read_spark_data_frame(file_name):
    """

    :param file_name:
    :return:
    """
    data = pd.read_csv(file_name)
    spark_data = spark.createDataFrame(data)
    return spark_data


def save_predictions_xente(file_name, transactions_list, test_pred, id_list):
    """
    :param file_name:
    :param transactions_list:
    :param test_pred:
    :param id_list:
    :return:
    """
    file = open(file_name, 'w')
    file.write('TransactionId,FraudResult\n')
    for trans_id, value in zip(transactions_list, test_pred):
        file.write('{0},{1}\n'.format(trans_id, int(value)))
    for item in (set(transactions_list)-id_list):
        file.write('{0},0\n'.format(item))
    file.close()

