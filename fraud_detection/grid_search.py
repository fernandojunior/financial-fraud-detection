import os

# LSCP
lscp_bagging_n_estimators = [2, 4, 8, 16, 32]
lscp_lof_n_neighbors = [2, 4, 8, 16, 32]
lscp_cblof_n_clusters = [2, 4, 8, 16, 32]
# KNN
knn_num_neighbors = [2, 4, 8, 16, 32]
knn_methods = ['largest', 'mean', 'median']
# Catboost
catboost_depth = [2, 4, 8, 16, 32]
catboost_learning_rate = [0.001, 0.01]
catboost_l2_leaf_reg = [2, 4, 8, 16, 32]


# Start the grid_search
for lscp_n_estimators in lscp_bagging_n_estimators:
    for lscp_neighbors in lscp_lof_n_neighbors:
        for lscp_clusters in lscp_cblof_n_clusters:
            for knn_neighbors in knn_num_neighbors:
                for knn_method in knn_methods:
                    for cat_depth in catboost_depth:
                        for cat_lr in catboost_learning_rate:
                            for cat_l2 in catboost_l2_leaf_reg:
                                hyperparam_list = \
                                    (f'{lscp_n_estimators},'
                                     f'{lscp_neighbors},'
                                     f'{lscp_clusters},'
                                     f'{knn_neighbors},'
                                     f'{knn_method},'
                                     f'{cat_depth},'
                                     f'{cat_lr},'
                                     f'{cat_l2}')

                                command = \
                                    f'python3.6 main.py run' \
                                    f'--input_train_file' \
                                    f'../data/xente_fraud_detection_train.csv' \
                                    f'--input_test_file' \
                                    f'../data/xente_fraud_detection_test.csv' \
                                    f'--output_balanced_train_x_file' \
                                    f'../data/balanced_train_x.csv' \
                                    f'--output_balanced_train_y_file' \
                                    f'../data/balanced_train_y.csv' \
                                    f'--output_valid_x_file' \
                                    f'../data/valid_x.csv' \
                                    f'--output_valid_y_file' \
                                    f'../data/valid_y.csv' \
                                    f'--output_valid_result_file' \
                                    f'../data/validation_{hyperparam_list}.csv' \
                                    f'--output_test_result_file' \
                                    f'../data/test_predictions_{hyperparam_list}.txt' \
                                    f'--lscp_bag_num_of_estimators {lscp_n_estimators}' \
                                    f'--lscp_lof_num_neighbors {lscp_neighbors}' \
                                    f'--lscp_cblof_num_clusters {lscp_clusters}' \
                                    f'--knn_num_neighbors {knn_neighbors}' \
                                    f'--knn_method {knn_method}' \
                                    f'--catboost_depth {cat_depth}' \
                                    f'--catboost_learning_rate {cat_lr}' \
                                    f'--catboost_l2_leaf_reg {cat_l2}'

                                os.system(command)
