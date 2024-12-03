import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
temperature=[20,25,30,35,40]
ice_cream_sales=[13,21,25,35,38]
x=np.array([temperature]).T
y=np.array(ice_cream_sales)
rmodel=LinearRegression()
rmodel=rmodel.fit(x,y)
rmodel_slope=rmodel.coef_
rmodel_intercept=rmodel.intercept_
print("MODEL SLOPE",rmodel_slope)
print("MODEL INTERCEPT",rmodel_intercept)
y_predict=rmodel.predict(x)
rmse=np.sqrt( mean_squared_error(y,y_predict))
r2=rmodel.score(x,y)
print("MODEL RMSE",rmse)
print("R SQUARED ERROR",r2)
plt.scatter(temperature,ice_cream_sales,marker='*',edgecolors='r')
plt.plot(temperature,y_predict,"-bo") 

