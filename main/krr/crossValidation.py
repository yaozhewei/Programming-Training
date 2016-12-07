import numpy
import sys
sys.path.append('../../main/kernel/')
import rbfKernelApprox
'''
def uniformSampling(lower, upper, size):
	arr = numpy.random.rand(size[0],size[1])
	diff = upper - lower
	arr = arr * diff + lower
	return numpy.sort(arr)


print numpy.random.uniform(0,3,[5,3]) # using this one to replace 
'''

def crossValid(matXtrain, vecYtrain, numFeature, parameters):

	n = matXtrain.shape[0]

	# seperate data into partitions 

	foldofCrossValid = 5 # cen be tuned
	randperm = numpy.random.permutation(n)
	nBegin = 0
	nstep = int( numpy.ceil( n / foldofCrossValid  ) ) + 1
	idxTrainValidData = list()
	for i in range(foldofCrossValid):
		nEnd = min( nBegin + nstep, n )
		idx = randperm[nBegin:nEnd]
		idxTrainValidData.append(idx)
		nBegin = nEnd

	listVecY = list()
	for i in range(foldofCrossValid):
		idx = idxTrainValidData[i]
		listVecY.append(vecYtrain[idx, :])

	# randomly get parameters

	sigmaRange = numpy.random.uniform( parameters['sigmaLower'], parameters['sigmaUpper'], [parameters['sigmaNum'], 1 ] )
	gammaRange = numpy.random.uniform( parameters['gammaLower'], parameters['gammaUpper'], [parameters['sigmaNum'], parameters['gammaNum'] ] )
	gammaRange = numpy.exp(gammaRange)

	# cross validation
	matMSE = numpy.zeros( (parameters['sigmaNum'], parameters['gammaNum']) )

	for i in range(parameters['sigmaNum']):
		sigma = sigmaRange[i]

		if parameters['method'] == 'Nystrom':
			matUL, vecSL = rbfKernelApprox.nystrom(matXtrain, sigma, numFeature)
		if  parameters['method'] == 'RandomFeature':
			matUL, vecSL = rbfKernelApprox.RandomFeature(matXtrain, sigma, numFeature)

		listMatUL = list()
		for fold in range(foldofCrossValid):
			idx = idxTrainValidData[fold]
			listMatUL.append( matUL[idx, :] )

		del matUL
		#  find best Gamma for fixed sigma
		arrGamma = gammaRange[i, :]
		arrMSE = crossValidGamma(listMatUL, vecSL, listVecY, arrGamma)
		matMSE[i, :] = arrMSE


	# find best sigma and gamma
	idx = matMSE.argmin()
	idx0 = int(numpy.floor( idx / parameters['gammaNum'] ))
	idx1 = idx % parameters['gammaNum']
	sigmaOpt = sigmaRange[idx0, 0]
	gammaOpt = gammaRange[idx0, idx1]
	mse = matMSE[idx0, idx1]

	return sigmaOpt, gammaOpt, mse

def crossValidGamma(listMatUL, vecSL, listVecY, arrGamma):
	foldofCrossValid = len(listMatUL)
	numGamma = arrGamma.shape[0]
	arrMSE = numpy.zeros(numGamma)

	d = listMatUL[0].shape[1]
	n = 0

	for vecY in listVecY:
		n += vecY.shape[0]

	listMatU = list()
	listVecUy = list()
	listVecSsq = list()

	for fold in range(foldofCrossValid):
		matU, vecS = numpy.linalg.svd( listMatUL[fold] * vecSL.reshape(1,d), full_matrices = False ) [0:2]

		vecSsq = vecS * vecS

		vecUy = numpy.dot(matU.T, listVecY[fold]) 

		listVecSsq.append(vecSsq.reshape(d, 1))
		listMatU.append( matU )
		listVecUy.append( vecUy )

	del matU
	del vecSsq
	del vecUy
	del vecS

	vecSLsq = (vecSL * vecSL ).reshape(d, 1 )

	for j in range(numGamma):
		gamma = arrGamma[j]

		listModel = list()

		for fold in range(foldofCrossValid):

			gammaScaled = gamma * listMatUL[fold].shape[0]
			# train
			model = listVecUy[fold] / (gammaScaled / listVecSsq[fold] + 1)
			model = numpy.dot(listMatU[fold], model)
			model = listVecY[fold] - model
			model = model / gammaScaled

			model = numpy.dot( listMatUL[fold].T, model )
			model = model * vecSLsq
			listModel.append(model)
		del model

		# validation
		squaredErr = 0
		for fold in range(foldofCrossValid):
			for foldValid in range(foldofCrossValid):
				if foldValid != fold:
					vecYpredict = numpy.dot( listMatUL[foldValid], listModel[fold] )
					err = numpy.linalg.norm( listVecY[foldValid] - vecYpredict )
					squaredErr += err * err

		arrMSE[j] = squaredErr / (foldofCrossValid - 1) / n

	return arrMSE 


'''
matXtrain = numpy.random.randn(10000,100)
w = numpy.random.randn(100,1)
VecYtrain = numpy.dot(matXtrain, w)

s = 50;

parameters = {'method' : 'Nystrom', 'sigmaLower' : 1, 'sigmaUpper' : 10, 'sigmaNum' : 5, 'gammaLower': -20, 'gammaUpper': -10, 'gammaNum': 20}
print crossValid(matXtrain, VecYtrain, s, parameters)
'''























