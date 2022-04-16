import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn import metrics
import time
import sklearn.metrics as sm
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
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
data['MasVnrType'] = data['MasVnrType'].fillna(value = 'None', method = None)
data['BsmtQual'] = data['BsmtQual'].fillna(value = 'None', method = None)
data['BsmtCond'] = data['BsmtCond'].fillna(value = 'None', method = None)
data['BsmtExposure'] = data['BsmtExposure'].fillna(value = 'None', method = None)
data['BsmtFinType1'] = data['BsmtFinType1'].fillna(value = 'None', method = None)
data['BsmtFinType2'] = data['BsmtFinType2'].fillna(value = 'None', method = None)
# data[ColName] = data[ColName].fillna(value='None', method=None)
X = Feature_Encoder(X, columns)
# obj = data.isna().sum()
# for key,value in obj.iteritems():
#     if value > 0:
#         print(key,"   ",value)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=True)
corr = house_data.corr()
top_feature = corr.index[abs(corr['SalePrice'] > 0.5)]
plt.subplots(figsize=(12, 8))
top_corr = house_data[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()
# top_feature = top_feature.delete(-1)
# X = X[top_feature]
# X = featureScaling(X,0,1)
poly_features = PolynomialFeatures(degree = 3)
X_train_poly = poly_features.fit_transform(X_train)
# sc_X = StandardScaler()
# X_train_poly = sc_X.fit_transform(X_train_poly)
# poly_model = linear_model.LinearRegression()
poly_model = linear_model.Ridge(alpha=.000001, max_iter=2000)
start = time.time()
poly_model.fit(X_train_poly, Y_train)
stop = time.time()
Y_train_predicted = poly_model.predict(X_train_poly)
prediction = poly_model.predict(poly_features.fit_transform(X_test))
print('Co-efficient of linear regression', poly_model.coef_)
print('Intercept of linear regression model', poly_model.intercept_)
print('Mean Square Error', metrics.mean_squared_error(Y_test, prediction))
print('R2 Score', sm.r2_score(Y_test, prediction))
print(f"Training time: {stop - start}s")
