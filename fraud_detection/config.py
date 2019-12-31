random_seed = 42
num_jobs = 12
device_type = 'GPU'

# Timestamp (yyyy/mm/dd HH:MM:SS)
feature_column_timestamp = 'TransactionStartTime'
# Transaction Value
feature_column_value = ['Value']
# Select Categorical columns to check Value spent over time
feature_categorical_to_check_spent_value = ['ProviderId',
                                            'ProductId']
# Others Categorical columns
others_categorical_columns_list = ['ProductCategory',
                                   'ChannelId',
                                   'PricingStrategy',
                                   'TransactionId',
                                   'BatchId']
# Others Numerical columns
others_numerical_columns_list = []
# Target outcome column
target_column_name = ['FraudResult']
