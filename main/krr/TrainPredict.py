import numpy
import scipy.cluster.vq as scvq
import sys
sys.path.append('../../main/kernel/')
import rbfKernelApprox

def TrainPredict(matXtrain, VecYtrain, matXtest, VecYtest, numFeature, sigmaOpt, gammaOpt, parameters):

	# prepare
	n = matXtrain.shape[0]
	m = matXtest.shape[0]
	matX = numpy.concatenate( (matXtrain, matXtest) )
	del matXtrain
	del matXtest


	# extract feature 
	if parameters['method'] == 'Nystrom':
		matUL, vecSL = rbfKernelApprox.nystrom(matX, sigmaOpt, numFeature)
	if parameters['method'] == 'RandomFeature':
		matUL, vecSL = rbfKernelApprox.RandomFeature(matX, sigmaOpt, numFeature)

	del matX

	d = matUL.shape[1]
	matULtrain = matUL[:n, :]
	matULtest = matUL[n:,:]

	del matUL

	#train
	gammaScaled = gammaOpt * n

	matU, vecS = numpy.linalg.svd( matULtrain * vecSL.reshape(1, d), full_matrices = False )[0:2]
	model = numpy.dot(matU.T, VecYtrain
)
	model = model / (gammaScaled / (vecS * vecS) + 1 ).reshape(d, 1)
	model = numpy.dot(matU, model)
	model = VecYtrain - model
	model = model / gammaScaled

	del matU
	del vecS

	#prediction
	vecYpredict = numpy.dot(matULtrain.T, model)
	del matULtrain
	vecYpredict = vecYpredict * (vecSL * vecSL).reshape(d, 1)
	vecYpredict = numpy.dot( matULtest, vecYpredict )

	# error
	err = numpy.linalg.norm(VecYtest - vecYpredict)

	return err * err / m

'''
matXtrain = numpy.random.randn(100,10)
matXtest = numpy.random.randn(30,10)
w = numpy.random.randn(10,1)
VecYtrain = numpy.dot(matXtrain, w)
VecYtest = numpy.dot(matXtest, w)

s = 50;
sigma = 1
gamma = 12
parameters = {'method' : 'Nystrom'}
print TrainPredict(matXtrain, VecYtrain, matXtest, VecYtest, s, sigma, gamma, parameters)
'''









