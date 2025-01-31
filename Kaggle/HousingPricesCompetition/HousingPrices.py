import numpy as np
import pandas as pd
import random
import csv
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
import time 
start_time = time.time()

train_data = pd.read_csv("D:\Ameri\Kaggle\HousingPricesCompetition\Judging\inputTrain.csv")
test_data = pd.read_csv("D:\Ameri\Kaggle\HousingPricesCompetition\Judging\inputTest.csv")
original_data = train_data
features = train_data.drop('SalePrice', axis = 1).columns

# Pre-processing
for category in features:
    cntNA = False
    for x in train_data[category]:
        if (pd.isna(x)):
            cntNA = True
            break
    if (cntNA == True): 
        features = features.drop(category)

encoded_train_data = train_data
encoded_test_data = test_data
for category in features:
    if (isinstance(train_data[category][0], np.int64)): continue
    encoded_train_data[category] = LabelEncoder().fit_transform(encoded_train_data[category].tolist())
    encoded_test_data[category] = LabelEncoder().fit_transform(encoded_test_data[category].tolist())

result = []
for i in range(test_data.shape[0]): result.append(1000000000)
for i in range(1):
    X_train = encoded_train_data[features].values
    y_train = original_data['SalePrice']

    forest_model = RandomForestRegressor()
    forest_model.fit(X_train, y_train)

    for j in range(test_data.shape[0]):
        row = encoded_test_data.iloc[j][features].tolist()
        result[j] = min(result[j], forest_model.predict(np.array(row).reshape(1, -1))[0])

output = []
output.append(['Id', 'SalePrice'])
for i in range(test_data.shape[0]):
    value = [str(test_data['Id'][i]), str(result[i])]
    output.append(value)

filename = 'D:\Ameri\Kaggle\HousingPricesCompetition\submission.csv'
with open(filename, 'w', newline = '') as file:
    writer = csv.writer(file)
    writer.writerows(output)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Execution time: {elapsed_time:.2f} seconds")
