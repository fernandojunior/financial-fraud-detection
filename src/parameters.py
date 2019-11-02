import os
import socket
import sys


def get_java_home():
    if socket.gethostname() == "Italo":
        path = "/usr/lib/jvm/java-8-openjdk-amd64"
    elif socket.gethostname() == "dunfrey-AERO-15-X9":
        path = "/usr/lib/jvm/java-8-openjdk-amd64"
    else:
        print('Please, specify manually the environment in the main code.')
        sys.exit()
    return path


def get_spark_home():
    if socket.gethostname() == "Italo":
        path = "/media/workspace/install/spark-2.4.4-bin-hadoop2.7"
    elif socket.gethostname() == "dunfrey-AERO-15-X9":
        path = "/home/dunfrey/spark-2.4.3-bin-hadoop2.7"
    else:
        print('Please, specify manually the environment in the main code.')
        sys.exit()
    return path


def get_file_name(type_file):
    if type_file == 'training_data':
        file_name = '../data/xente_fraud_detection_train.csv'
        if not os.path.isfile(file_name):
            file_name = 'https://drive.google.com/uc?export=download&id=1NrtVkKv8n_g27w5elq9HWZA1i8aFBW0G'
    elif type_file == 'test_data':
        file_name = '../data/xente_fraud_detection_test.csv'
        if not os.path.isfile(file_name):
            file_name = 'https://drive.google.com/uc?export=download&id=16cRQIFW6n2th2YOK7DEsp9dQgihHDuHa'
    else:
        print('Please, choose a valid link name.')
        sys.exit()
    return file_name

