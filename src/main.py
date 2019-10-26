import socket
import os
import sys

if socket.gethostname() == "Italo":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/media/workspace/install/spark-2.4.4-bin-hadoop2.7"
elif socket.gethostname() == "dunfrey-AERO-15-X9":
    os.environ["JAVA_HOME"] = "/usr/lib/jvm/java-8-openjdk-amd64"
    os.environ["SPARK_HOME"] = "/home/dunfrey/spark-2.4.3-bin-hadoop2.7"
else:
    print('Please, specify manually the environment in the main code.')
    sys.exit()

import findspark
findspark.init()
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName('xente').getOrCreate()

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pyspark.sql.functions as F

from pyspark.sql.types import IntegerType
from pyspark.ml.feature import StringIndexer
from pyspark.sql.functions import mean
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from imblearn.over_sampling import SMOTENC


def read_data_from_web(url):
    data = pd.read_csv(url)
    spark_data = spark.createDataFrame(data)
    return spark_data


fraud_data = read_data_from_web("https://drive.google.com/uc?export=download&id=1NrtVkKv8n_g27w5elq9HWZA1i8aFBW0G")
print(fraud_data.show())







