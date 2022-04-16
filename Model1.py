import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn import linear_model
from sklearn import metrics
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from Pre_processing import *
data = pd.read_csv('House_Data.csv')
data.dropna(axis = 1, how = 'all', inplace = True)
data = data.fillna(data.mean())
# data1 = data[['LotArea', 'Street', 'MSSubClass', 'MasVnrType', 'SalePrice']]
# data1.dropna(how = 'any', inplace = True)
house_data = data.iloc[:,:]
X = data.iloc[:,1:45]
Y = data["SalePrice"]
columns = ('Street', 'LotShape', 'Utilities', 'MSZoning', 'LotConfig', 'LandSlope', 'Neighborhood', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical')
# columns = ('Street', 'MasVnrType')
print(data.describe())
X = Feature_Encoder(X, columns)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True, random_state=10)
corr = house_data.corr()
top_feature = corr.index[abs(corr['SalePrice'] > 0.5)]
plt.subplots(figsize=(12, 8))
top_corr = house_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
# top_feature = top_feature.delete(-1)
# X = X[top_feature]
# X = featureScaling(X,0,1)
cls = linear_model.LinearRegression()
start = time.time()
cls.fit(X_train, Y_train)
stop = time.time()
prediction = cls.predict(X_test)
print('Co-efficient of linear regression', cls.coef_)
print('Intercept of linear regression model', cls.intercept_)
print('Mean Square Error', metrics.mean_squared_error(np.asarray(Y_test), prediction))
print('R2 Score', sm.r2_score(Y_test, prediction))
print(f"Training time: {stop - start}s")
