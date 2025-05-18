import loadData
import numpy as np
import modelGenerationAndTraining as mg


dataTrain, inputTest, outputTest = loadData.loadData()
X_train = np.transpose([dataTrain[:,3]])
Y_train = dataTrain[:,0]
X_test = np.transpose([inputTest[:,2]])
Y_test = np.transpose(outputTest)[0]



linRegModel = mg.trainLinreg(X_train, Y_train)
mg.test(linRegModel, X_test, Y_test)



