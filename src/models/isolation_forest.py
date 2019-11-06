import pandas as pd
import sklearn.ensemble as ens
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
    def __init__(self, max_sample=100, contamination=0.01, test_proportion=0.3, random_value=42):
        '''
        :param max_sample: The number of samples to draw from X to train each base estimator.
        :param contamination: The amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold on the decision
        function. If ‘auto’, the decision function threshold is determined as in the original paper.
        More details are available here:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        :param test_proportion: If float, should be between 0.0 and 1.0 and represent the proportion
        of the dataset to include in the test split. If int, represents the absolute number of test
        samples. If None, the value is set to the complement of the train size. If train_size is also None,
        it will be set to 0.25.
        More details are available here:
        https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
        :param random_value: Pseudo-random number generator.
        More details are available here:
        https://docs.scipy.org/doc/numpy-1.16.0/reference/generated/numpy.random.RandomState.html
        '''
        self.graph_performance = pd.DataFrame([], columns=['score', 'max', 'contamination'])
        self.test_proportion = test_proportion
        self.max_sample = max_sample
        self.contamination = contamination
        self.random_value = random_value
        self.clf = ens.IsolationForest(behaviour='new', max_samples=self.max_sample,
                                       random_state=self.random_value, contamination=self.contamination)

    def evaluate_model(self, model, x_test, y_test, metric='f1score'):
        '''
        :param model: The model trained used to be evaluated.
        :param x_test:
        :param y_test:
        :param metric: Could be any metric implemented in statistic module, such as
        PPV, TPR, FPR, or F1score.
        :return:
        '''
        y_predictions_test = model.predict(x_test)
        y_predictions_test = norm_pred(y_predictions_test)
        cm = confusion_matrix(y_test, y_predictions_test)
        score = stat.compute_score(cm, metric)
        return score

    def fit_grid_search(self, training_data, testing_data, max_sample_list, contamination_list, metric='f1score'):
        '''
        :param training_data:
        :param testing_data:
        :param max_sample_list: A list with the number of samples to draw from X to train each base estimator.
        :param contamination_list: A list with the amount of contamination of the data set, i.e. the proportion
        of outliers in the data set. Used when fitting to define the threshold on the decision
        function. If ‘auto’, the decision function threshold is determined as in the original paper.
        More details are available here:
        https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html
        :param metric: Performance evaluator used to estimate the model performance.
        :return: 5 variables, they are: the model trained, the best tuple of parameters (max_sample and contamination),
        and graph_performance which contains the performance model for each tuple hyperparameter combination.
        '''
        x_train, x_test, y_train, y_test = \
            train_test_split(training_data, testing_data,
                             test_size=self.test_proportion, random_state=self.random_value)

        baseline = 0.0
        self.graph_performance = pd.DataFrame([], columns=['score', 'max', 'contamination'])
        for max_sample in max_sample_list:
            for contamination_level in contamination_list:
                model = ens.IsolationForest(behaviour='new', max_samples=max_sample,
                                            random_state=self.random_value, contamination=contamination_level)
                model.fit(x_train)
                score = self.evaluate_model(model, x_test, y_test, metric)
                if score > baseline:
                    baseline = score
                    self.clf = model
                    self.max_sample = max_sample
                    self.contamination = contamination_level
                performance = pd.DataFrame({'score': [score], 'max': [max_sample], 'contamination': [contamination_level]})
                self.graph_performance = self.graph_performance.append(performance)
        return self.clf, baseline, self.max_sample, self.contamination, self.graph_performance

    def fit(self, x_train):
        '''
        Fit the Isolation Forest in the training data.
        :param: x_train:
        :return: Trained model
        '''
        if x_train:
            self.clf.fit(x_train)
        return self.clf
