import numpy
import sys
sys.path.append('../../main/regression')
import regression
import scipy.io 

datafn = '../../data/compressdata.mat'
dataDict = scipy.io.loadmat(datafn)


matXtrain = dataDict['Xtrain']
matXtest = dataDict['Xtest']
vecYtrain = dataDict['ytrain']
vecYtest = dataDict['ytest']

n, d= matXtrain.shape


m = matXtest.shape[0]


del dataDict

matXY = numpy.dot(matXtrain.T, vecYtrain)
matXX = numpy.dot(matXtrain.T, matXtrain)

gamLower = -29
gamUpper = 2
gamNum = 500

gamma = numpy.random.uniform(gamLower,gamUpper,[gamNum,1])
MSEs = numpy.zeros((gamNum, 1))

for i in range(gamNum):
	gammascaled = n * numpy.exp( gamma[i,0] )
	matXXgamma = numpy.linalg.inv( matXX + gammascaled * numpy.eye(d) )
	W = numpy.dot(matXXgamma, matXY)
	vecYpredict = numpy.dot(matXtest, W)
	err = numpy.linalg.norm(vecYtest - vecYpredict)
	MSEs[i,0] = err * err / m

outputFileName = 'YearPrediction' + '.mat'

resultDict = {'gamma': gamma, 'MSEs':MSEs}

scipy.io.savemat(outputFileName, resultDict)
