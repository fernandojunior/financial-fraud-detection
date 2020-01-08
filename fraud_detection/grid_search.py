import os
import sklearn.model_selection


param_grid = {
    # LSCP
    'lscp_bagging_n_estimators': [2, 4, 8, 16, 32],
    'lscp_lof_n_neighbors': [2, 4, 8, 16, 32],
    'lscp_cblof_n_clusters': [2, 4, 8, 16, 32],
    # Isolation Forest
    'iso_forest_estimators': [2, 4, 8, 16, 32],
    # KNN
    'knn_num_neighbors': [2, 4, 8, 16, 32],
    'knn_methods': ['largest', 'mean', 'median'],
    # Catboost
    'catboost_depth': [2, 4, 8, 16, 32],
    'catboost_learning_rate': [0.0003, 0.001, 0.01],
    'catboost_l2_leaf_reg': [2, 4, 8, 16, 32],
    'catboost_number_of_iterations': [100, 400, 800]
}


def make_iteration(is_estimators,
                   lscp_n_estimators,
                   lscp_neighbors,
                   lscp_clusters,
                   knn_neighbors,
                   knn_method,
                   cat_depth,
                   cat_lr,
                   cat_l2,
                   cat_iter):
    hyperparam_list = \
        (f'{is_estimators},'
         f'{lscp_n_estimators},'
         f'{lscp_neighbors},'
         f'{lscp_clusters},'
         f'{knn_neighbors},'
         f'{knn_method},'
         f'{cat_depth},'
         f'{cat_lr},'
         f'{cat_l2},'
         f'{cat_iter}')
    print(f'Running a new iteration with the params: {hyperparam_list}')

    command = \
        f'python3.6 main.py run ' \
        f'--input_train_file ' \
        f'../data/xente_fraud_detection_train.csv ' \
        f'--input_test_file ' \
        f'../data/xente_fraud_detection_test.csv ' \
        f'--output_balanced_train_x_file ' \
        f'../data/balanced_train_x.csv ' \
        f'--output_balanced_train_y_file ' \
        f'../data/balanced_train_y.csv ' \
        f'--output_valid_x_file ' \
        f'../data/valid_x.csv ' \
        f'--output_valid_y_file ' \
        f'../data/valid_y.csv ' \
        f'--output_valid_result_file ' \
        f'../data/validation_{hyperparam_list}.csv ' \
        f'--output_test_result_file ' \
        f'../data/test_predictions_{hyperparam_list}.txt ' \
        f'--isolation_forest_num_estimators {is_estimators} ' \
        f'--lscp_bag_num_of_estimators {lscp_n_estimators} ' \
        f'--lscp_lof_num_neighbors {lscp_neighbors} ' \
        f'--lscp_cblof_num_clusters {lscp_clusters} ' \
        f'--knn_num_neighbors {knn_neighbors} ' \
        f'--knn_method {knn_method} ' \
        f'--catboost_depth {cat_depth} ' \
        f'--catboost_learning_rate {cat_lr} ' \
        f'--catboost_l2_leaf_reg {cat_l2} ' \
        f'--catboost_num_iterations {cat_iter}'

    print(f'Running the command: {command}')
    os.system(command)

    os.system('rm ../data/balanced_train_x.csv')
    os.system('rm ../data/balanced_train_y.csv')
    os.system('rm ../data/valid_x.csv')
    os.system('rm ../data/valid_y.csv')

    os.system('rm ../data/catBoost_model')
    os.system('rm ../data/model_if')
    os.system('rm ../data/model_knn')
    os.system('rm ../data/model_lscp')


# Start the grid_search
for param_combination in sklearn.model_selection.ParameterGrid(param_grid):
    make_iteration(
        param_combination['is_estimators'],
        param_combination['lscp_n_estimators'],
        param_combination['lscp_neighbors'],
        param_combination['lscp_clusters'],
        param_combination['knn_neighbors'],
        param_combination['knn_method'],
        param_combination['cat_depth'],
        param_combination['cat_lr'],
        param_combination['cat_l2'],
        param_combination['cat_iter'])
