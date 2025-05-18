import sklearn as skl
import sklearn.linear_model as lm
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

"""

# Linear Regression
def trainLinreg(X_train, Y_train):
    # Create linear regression object
    linRegModel = lm.LinearRegression()
    # Train the model using the training sets
    linRegModel.fit(X_train, Y_train)
    # print(linRegModel.coef_)
    return linRegModel

# k-nearest neighbor (kNN)
def trainKNNReg(X_train, Y_train):
    # Create k-nearest neighbor regression object
    KNNRegModel = KNeighborsRegressor()
    # Train the model using the training sets
    KNNRegModel.fit(X_train, Y_train)
    # print(KNNRegModel.coef_)
    return KNNRegModel

# supported vector regression (SVR)
def trainSVR(X_train, Y_train):
    # Create support vector regression object
    SuVeRegModel = svm.SVR()
    # Train the model using the training sets
    SuVeRegModel.fit(X_train, Y_train)
    # print(SuVeRegModel.coef_)
    return SuVeRegModel

# artificial neural networks (ANN)
def trainANN(X_train, Y_train):
    # Create k-nearest neighbor regression object
    ANNModel = MLPRegressor()
    # Train the model using the training sets
    ANNModel.fit(X_train, Y_train)
    # print(ANNModel.coef_)
    return ANNModel

# prediction and evaluation through RMSE
def test(model, X_test, Y_test):
    # Make predictions using the testing set (Linear Regression)
    Y_pred = model.predict(X_test)
    # calculate RMSE
    RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
    #print(RMSE)
    return RMSE, Y_pred
"""
import sklearn as skl
import sklearn.linear_model as lm
from sklearn import svm
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import numpy as np



# Linear Regression
def trainLinreg(X_train, Y_train):
    # Create linear regression object
    linRegModel = lm.LinearRegression()
    # Train the model using the training sets
    linRegModel.fit(X_train.reshape(-1, 1), Y_train)
    # print(linRegModel.coef_)
    return linRegModel

# MultiLinear Regression
def trainLinreg2(X_train, Y_train):
    # Create linear regression object
    linRegModel = lm.LinearRegression()
    # Train the model using the training sets
    linRegModel.fit(X_train, Y_train)
    # print(linRegModel.coef_)
    return linRegModel

# k-nearest neighbor (kNN)
def trainKNNReg(X_train, Y_train):
    # Create k-nearest neighbor regression object
    KNNRegModel = KNeighborsRegressor()
    # Train the model using the training sets
    KNNRegModel.fit(X_train.reshape(-1, 1), Y_train)
    # print(KNNRegModel.coef_)
    return KNNRegModel

# support vector regression (SVR)
def trainSVR(X_train, Y_train):
    # Create support vector regression object
    SuVeRegModel = svm.SVR()
    # Train the model using the training sets
    SuVeRegModel.fit(X_train.reshape(-1, 1), Y_train)
    # print(SuVeRegModel.coef_)
    return SuVeRegModel

def trainSVR2(X_train, Y_train):
    # Create support vector regression object
    SuVeRegModel = svm.SVR()
    # Train the model using the training sets
    SuVeRegModel.fit(X_train.reshape, Y_train)
    # print(SuVeRegModel.coef_)
    return SuVeRegModel

# artificial neural networks (ANN)
def trainANN(X_train, Y_train):
    # Create k-nearest neighbor regression object
    ANNModel = MLPRegressor()
    # Train the model using the training sets
    ANNModel.fit(X_train.reshape(-1, 1), Y_train)
    # print(ANNModel.coef_)
    return ANNModel

# artificial neural networks (ANN)
def trainANN2(X_train, Y_train):
    # Create k-nearest neighbor regression object
    ANNModel = MLPRegressor()
    # Train the model using the training sets
    ANNModel.fit(X_train, Y_train)
    # print(ANNModel.coef_)
    return ANNModel

# prediction and evaluation through RMSE
def test(model, X_test, Y_test):
    # Make predictions using the testing set (Linear Regression)
    Y_pred = model.predict(X_test.reshape(-1, 1))
    # calculate RMSE
    RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return RMSE, Y_pred

# prediction and evaluation through RMSE
def test2(model, X_test, Y_test):
    # Make predictions using the testing set (Linear Regression)
    Y_pred = model.predict(X_test)
    # calculate RMSE
    RMSE = np.sqrt(mean_squared_error(Y_test, Y_pred))
    return RMSE, Y_pred