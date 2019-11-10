from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split, GridSearchCV

import sys
sys.path.insert(0, 'models/')

import balance as bal
import isolation_forest
import parameters as param
import preprocessing as preprocess
import visualization as vis
import statistics as stats
import io_module


# This param when defined as True will execute the complete code, so slowly processing time
# because is require to execute all checks and print all not essential functions.
# When defined as False, a fast processing is applied with all core functionalities working
# well.
full_execution = False
verbose_mode = True

all_features = ['ProductId', 'ProductCategory', 'ChannelId', 'Value', 'PricingStrategy', 'Operation', 'PositiveAmount',
                'avg_ps_ChannelId', 'rt_avg_ps_ChannelId', 'avg_ps_ProductCategory', 'rt_avg_ps_ProductCategory',
                'avg_ps_ProductId', 'rt_avg_ps_ProductId']
columns_to_remove = ['CurrencyCode', 'CountryCode', 'BatchId', 'AccountId', 'SubscriptionId',
                     'CustomerId', 'TransactionStartTime', 'Amount']
categorical_features = ['ProductId', 'ProductCategory', 'ChannelId']
numerical_features = ['PositiveAmount', 'Operation', 'Value', 'PricingStrategy']
numerical_features_augmented = ['Value', 'PricingStrategy', 'Operation', 'PositiveAmount', 'avg_ps_ChannelId',
                                'rt_avg_ps_ChannelId', 'avg_ps_ProductCategory', 'rt_avg_ps_ProductCategory',
                                'avg_ps_ProductId', 'rt_avg_ps_ProductId']

label = 'FraudResult'
genuine_label = 'FraudResult==0'
fraud_label = 'FraudResult==1'
categorical_positions = [0, 1, 2, 4]


# Read Fraud Detection Challenge data
train_data = io_module.read_spark_data_frame(param.get_file_name('training_data'))

# Create new features and remove the non used features
train_data = preprocess.get_features_augmentation(train_data)
train_data = train_data.drop(*columns_to_remove)

# Print Description details about the data set
if full_execution:
    # Checking if there are missing data or duplicate line?
    preprocess.there_is_missing_data(train_data)
    preprocess.there_is_duplicate_lines(train_data)
    # Plot transactions proportion comparing fraudulent with genuine transactions
    vis.plot_transactions_proportions(train_data, "ProductCategory")
    vis.plot_transactions_proportions(train_data, "ChannelId")
    # Print a full description over the data
    stats.print_description(train_data, numerical_features)
    # Plot histogram distribution for all features
    # True is used for genuine data
    # False is used for fraudulent data
    vis.plot_hist(train_data, numerical_features, True)
    vis.plot_hist(train_data, numerical_features, False)
    # Plot correlation matrix for fraudulent and genuine data
    vis.plot_heatmap(train_data, numerical_features_augmented, True)
    vis.plot_heatmap(train_data, numerical_features_augmented, False)

    vis.plot_bar((train_data.toPandas())[label], 'Genuine and Fraud transactions')
    train_data_smotenc_x, train_data_smotenc_y = bal.balance_using_smotenc(train_data, all_features, label,
                                                                           categorical_positions)
    vis.plot_bar(train_data_smotenc_y[label], 'Genuine and Fraud transactions')

    for oversampling_technique in ['smote', 'adasyn', 'random']:
        model = isolation_forest.IsolationForest()
        max_sample_list = [200, 500, 1000, 5000, 10000, 15000, 25000]
        contamination_list = [0.5]
        x_data, y_data = bal.balance_using_only_numerical_features(train_data, numerical_features,
                                                                   label, oversampling_technique)
        graph_performance = model.fit_grid_search(x_data, y_data, max_sample_list, contamination_list, 'accuracy')
        vis.plot_performance_comparison(graph_performance)




x_data, y_data = bal.balance_using_only_numerical_features(train_data, numerical_features, label, 'random')
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.3)
params = {'depth': [3, 4, 5],
                  'loss_function': ['Logloss'],
                  'eval_metric': ['AUC', 'F1'],
                  'iterations': [30, 50, 100, 150]}

model = CatBoostClassifier(eval_metric='AUC')
grid = GridSearchCV(estimator=model, param_grid=params, cv=3)
grid.fit(X_train, y_train)

print("\n The best score across ALL searched params:\n",
      grid.best_score_)

print("\n The best parameters across ALL searched params:\n",
      grid.best_params_)






# Evaluating with test data
"""
test_data = io_module.read_spark_data_frame(param.get_file_name('testing_data'))
test_data = preprocess.get_features_augmentation(test_data)
test_data = train_data.drop(*columns_to_remove)
"""

print('Finish with success')