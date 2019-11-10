import numpy as np
import pandas as pd
import sys
sys.path.insert(0, 'models/')

import cat_boost as cat

x_data = pd.read_csv('../data/fraud_data_smotenc_x.csv')
y_data = pd.read_csv('../data/fraud_data_smotenc_y.csv')
categorical_features = ['ProductId', 'ProductCategory', 'ChannelId']
categorical_positions = [0, 1, 2, 4]

model = cat.CatBoost()
model.fit(x_data, y_data, categorical_positions)


print('Finish with sucess')
#categorical_features