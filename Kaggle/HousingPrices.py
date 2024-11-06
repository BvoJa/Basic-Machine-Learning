import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv("D:\Download\melb_data.csv")
data = data.dropna(axis = 0)
features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea', 'YearBuilt', 'Lattitude', 'Longtitude']
X = data[features]
y = data.Price

from sklearn.metrics import mean_absolute_error
Model = DecisionTreeRegressor()
Model.fit(X, y)
print(mean_absolute_error(y, Model.predict(X)))