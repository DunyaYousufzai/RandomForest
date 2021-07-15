import numpy as np
import matplotlib.pyplot as plt
import pandas as  pd
from sklearn.ensemble import RandomForestRegressor

class random_forest:
    def __init__(self, file):
        self.file = file

    def data_selection(self, a, b,c):
       global rx, ry
       data = pd.read_csv(self.file)
       rx = data.iloc[:,a:b].values
       ry = data.iloc[:,c].values

    def trainig(self, k):
        global reg
        reg = RandomForestRegressor(n_estimators = k,random_state= 0)
        reg.fit(rx, ry)
    
    def plot(self, title,xlabel,ylabel):
        x_grid = np.arange(min(rx), max(rx), 0.01)
        x_grid = x_grid.reshape((len(x_grid), 1))
        plt.scatter(rx, ry, color = "green")
        plt.plot(x_grid, reg.predict(x_grid), color = "blue")
        plt.title("Random ForestRegressor")
        plt.xlabel('pos label')
        plt.xlabel('salary')
        plt.show()
    
    def predict(self, f):
        prediction_y = reg.predict([[f]])
        print(prediction_y)

poly = random_forest("random_forest.csv")
poly.data_selection(1,2,2)
poly.trainig(100)
poly.plot("polynomial Model", "position","salary")
poly.predict(8)