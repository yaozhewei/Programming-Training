import numpy
import time
import scipy.io 
import sys
sys.path.append('../../main/krr/')
import TrainPredict as tp 
import crossValidation as cv 

def krrExperiment(dataName, inputFileName, parameters, sketchSizes):
	# repeat crossValidation + TrainPredict
	numRepeat1 = 5
	# repeat fixed sigma and gamma, repeat TrainPredict
	numRepeat2 = 5

	dataDict =scipy.io.loadmat('../../data/' + inputFileName + '.mat')
	matXtrain = dataDict['Xtrain']
	matXtest = dataDict['Xtest']
	vecYtrain = dataDict['ytrain']
	vecYtest = dataDict['ytest']

	del dataDict

	numS = len(sketchSizes)

	mseMean = numpy.zeros(numS)

	MSEs = numpy.zeros( (numS, numRepeat1 * numRepeat2) )

	TimeCR = numpy.zeros(numS)
	TimeTP = numpy.zeros(numS)

	for l in range(numS):
		s=sketchSizes[l]
		for i in range(numRepeat1):
			t0 = time.time()
			sigmaOpt, gammaOpt, mseTmp = cv.crossValid(matXtrain, vecYtrain, s, parameters)
			t1 = time.time()
			mse = 0
			for j in range(numRepeat2):
				mse = tp.TrainPredict( matXtrain, vecYtrain, matXtest, vecYtest, s, sigmaOpt, gammaOpt, parameters )
				MSEs[l, i * numRepeat2 + j ] =mse
			t2 = time.time()

		mseMean[l] = numpy.mean( MSEs[l, :])
		TimeCR[l] = (t1 - t0) / numRepeat1
		TimeTP[l] =(t2 - t1) /numRepeat2 / numRepeat1

	outputFilename = dataName + '_' + parameters['method'] + '.mat'

	mathDict ={ 'MSEs':MSEs, 'mseMean':mseMean, 'sketchSizes':sketchSizes, 'TimeCR': TimeCR, 'TimeTP': TimeTP }

	scipy.io.savemat(outputFilename, mathDict)


dataName = 'YearPrediction'
inputFileName = 'compressdata'
#sketchSizes = [50, 70, 90, 110, 140, 170, 200, 240, 280, 320, 370, 420, 470, 530, 590, 650, 720, 790, 870, 950, 1040, 1140, 1250, 1370, 1500]
sketchSizes = [50, 100, 200, 400, 800,1500]

sigLower = 10
sigUpper = 70
sigNum = 10
gamLower = -29
gamUpper = -18
gamNum = 20

parameters = {'method': 'Nystrom', 'sigmaLower': sigLower, 'sigmaUpper': sigUpper, 'sigmaNum': sigNum, 'gammaLower': gamLower, 'gammaUpper': gamUpper, 'gammaNum': gamNum}
krrExperiment(dataName, inputFileName, parameters, sketchSizes)




