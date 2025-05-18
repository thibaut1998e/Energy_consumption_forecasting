import loadData
import numpy as np
import modelGenerationAndTraining as mg
import matplotlib.pyplot as plt
import csv


def load_column_from_csv(csv_file, column):
	return np.genfromtxt(csv_file, delimiter=',', dtype=None, skip_header=1, usecols=[column], encoding='UTF-8')
TIMESTAMPS_test = load_column_from_csv('Solution.csv', 0)
lableList = []
for x in range(30):
	temp = TIMESTAMPS_test[x*24][6:8]
	lableList.append((x*24, temp))

dataTrain, inputTest, outputTest = loadData.loadData()
powerTrain = dataTrain[:,0]
powerTest = np.transpose(outputTest)[0]

n = 10
X_train = np.array([powerTrain[k:k+n] for k in range(len(powerTrain)-n)])
Y_train = np.array([powerTrain[k] for k in range(n,len(powerTrain))])
X_test = np.array([powerTest[k:k+n] for k in range(len(powerTest)-n)])
Y_test = np.array([powerTest[k] for k in range(n,len(powerTest))])



LinRegModel = mg.trainLinreg2(X_train, Y_train)
RMSELinReg, Y_predLinReg = mg.test2(LinRegModel, X_test, Y_test)

#RMSE found : 0.12345
svrModel = mg.trainLinreg2(X_train, Y_train)
RMSEsvrModel, Y_predsvrModel = mg.test2(LinRegModel, X_test, Y_test)

#RMSE found : 0.12345
plt.plot(range(len(powerTest)), powerTest, label="real power")
plt.plot(range(n, len(powerTest)), Y_predLinReg, label = "predicted power by linear Regression")
plt.plot(range(n, len(powerTest)), Y_predsvrModel, label = "predicted power by svr model")

plt.xlabel("time")
plt.title("power prediction by linear regression and support vector machines models task 3")
plt.xticks(ticks=list(zip(*lableList))[0], labels=list(zip(*lableList))[1], rotation=90)
plt.legend()
#plt.legend(bbox_to_anchor=(0.15, 0.9, 0.8, 0), loc='lower left',ncol=2, mode="expand", borderaxespad=0.)
plt.show()



ANNModel = mg.trainANN2(X_train, Y_train)
RMSE, Y_pred = mg.test2(ANNModel, X_test, Y_test)

#RMSE Found : 0.122088
plt.plot(range(len(powerTest)), powerTest, label="real power")
plt.plot(range(n, len(powerTest)), Y_pred, label = "predicted power by ANN")
plt.xlabel("time")
plt.title("power prediction by ANN task 3 time series")
plt.xticks(ticks=list(zip(*lableList))[0], labels=list(zip(*lableList))[1], rotation=90)
plt.legend(bbox_to_anchor=(0.15, 0.9, 0.8, 0), loc='lower left',
                   ncol=2, mode="expand", borderaxespad=0.)
plt.show()






