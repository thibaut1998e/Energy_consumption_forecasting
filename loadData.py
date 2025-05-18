import numpy as np

def loadData():
    allData = []
    for fileName in ["TrainData.csv", "WeatherForecastInput.csv", "Solution.csv"]:
        file = open(fileName)
        data = file.read()
        data = data.split("\n")
        for i in range(len(data)):
            data[i] = data[i].split(",")
            data[i] = data[i][1:]

        data = data[1:len(data)-1]
        for i in range(len(data)):
            for j in range(len(data[0])):
                data[i][j] = float(data[i][j])
        allData.append(np.array(data))
    dataTrain, inputTest, outputTest = allData[0], allData[1], allData[2]
    return dataTrain, inputTest, outputTest


