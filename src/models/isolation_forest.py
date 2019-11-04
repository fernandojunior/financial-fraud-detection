import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import sys
sys.path.insert(0, '../')
import statistics as stat

def norm_pred(data):
    '''
    :param data: Pandas DataFrame or Numpy array with labels
        classified as -1 and 1, and you want to convert into
        1 and 0, respectively.
            [-1, 1]
    :return: [1, 0]
    '''
    data = ((data * -1) + 1) / 2
    return data


class IsolationForest:
    def __init__(self, numerical_features, label, outliers_label, test_proportion=0.3):
        self.graph_performance = pd.DataFrame([], columns=['score', 'max', 'cont'])
        self.numerical_features = numerical_features
        self.label = label
        self.outliers_label = outliers_label
        self.test_proportion = test_proportion

    def fit(self, training_data):
        fraud_data_pd_x = training_data.select(self.numerical_features).toPandas()
        fraud_data_pd_y = training_data.select(self.label).toPandas()

        self.X_train, self.X_test, self.y_train, self.y_test = \
                train_test_split(fraud_data_pd_x, fraud_data_pd_y,
                                 test_size=self.test_proportion,
                                 random_state=42)
        self.X_outliers = training_data.filter(self.outliers_label).select(self.numerical_features).toPandas()

        rng = np.random.RandomState(42)

        for max_sample in max_sample_list:
            for cont in cont_list:
                clf = IsolationForest(behaviour='new', max_samples=max_sample,
                                      random_state=rng, contamination=cont)

                clf.fit(X_train)

                y_pred_train = clf.predict(X_train)
                y_pred_test = clf.predict(X_test)
                y_pred_outliers = clf.predict(X_outliers)

                y_pred_test = norm_pred(y_pred_test)
                y_pred_outliers = norm_pred(y_pred_outliers)

                cm = confusion_matrix(y_test, y_pred_test)
                f1_score = stat.compute_score(cm)

                new_row = pd.DataFrame({'score': [f1_score], 'max': [max_sample], 'cont': [cont]})
                self.graph_performance = self.graph_performance.append(new_row)