from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV

import statistics as stats

class CatBoost:
    def __init__(self, depth=0.0, iterations=0.0, loss_function='Logloss', eval_metric='AUC'):
        self.performance_model = 0.0
        self.depth = depth
        self.iterations = iterations
        self.eval_metric = eval_metric
        self.loss_function = loss_function
        self.model = CatBoostClassifier(max_depth=depth, iterations=iterations)

    def define_best_param(self, params_dict):
        self.depth = params_dict['depth']
        self.eval_metric = params_dict['eval_metric']
        self.iterations = params_dict['iterations']
        self.loss_function = params_dict['loss_function']

    def fit_grid_search(self, x_data, y_data, num_folds=3, depth_list=[4, 8, 16], iterations_list=[100, 200, 300],
                        loss_function=['Logloss'], eval_metric_list=['AUC']):
        params = {'depth': depth_list,
                  'loss_function': loss_function,
                  'eval_metric': eval_metric_list,
                  'iterations': iterations_list}
        model = CatBoostClassifier(eval_metric=eval_metric_list)
        grid = GridSearchCV(estimator=model, param_grid=params, cv=num_folds)
        grid.fit(x_data, y_data)
        self.performance_model = grid.best_score_
        self.define_best_param(grid.best_params_)

    def fit(self, x_data, y_data, verbose=False, plot=False, categorical_features=[]):
        self.model.fit(x_data, y_data, verbose=verbose, plot=plot, cat_features=categorical_features)

    def predict(self, x_test):
        data = self.model.predict(x_test)
        return stats.norm_pred(data, 'inverse')