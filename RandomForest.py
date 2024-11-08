import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

def get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test):
    Model = DecisionTreeRegressor(max_leaf_nodes = max_leaf_nodes, random_state = 0)
    Model.fit(X_train, y_train)
    return mean_absolute_error(y_test, Model.predict(X_test))

data = pd.read_csv("D:\Download\melb_data.csv")
data = data.dropna(axis = 0)
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = data[features]
y = data.Price 
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, random_state = 0)

for max_leaf_nodes in [5, 50, 500, 5000, 50000]:
    mae = get_mae(max_leaf_nodes, X_train, X_test, y_train, y_test)
    print("Max leaf nodes: %d \t\t Mean Absolute Error: %f" %(max_leaf_nodes, mae))

forest_model = RandomForestRegressor(random_state = 0)
forest_model.fit(X_train, y_train)
mae = mean_absolute_error(y_test, forest_model.predict(X_test))
print("Mean Absolute Error using Random Forest: %f" %(mae))