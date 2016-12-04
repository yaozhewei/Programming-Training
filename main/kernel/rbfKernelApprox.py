import numpy
import kernelFun

def nystrom(matX, sigma, s):
	# calculate L, S of K = L * S * S * L
	k = int( numpy.ceil(s * 0.8) ) #for rank approx
	n = matX.shape[0]

	arrIndex = numpy.random.choice(n, s, replace = False)

	matC = kernelFun.rbfkernel(matX, matX[arrIndex, :], sigma )
	matW = matC[arrIndex, :] 
	matUW, vecSW, matVW = numpy.linalg.svd(matW, full_matrices = False)
	matUW = matUW[:, 0:k] / numpy.sqrt(vecSW[0:k])

	matC = numpy.dot(matC, matUW)
	matUL, vecSL, matVL = numpy.linalg.svd(matC, full_matrices = False)

	return matUL, vecSL

def RandomFeature(matX, sigma, s):
	# also calculate half part
	k = int( numpy.ceil(s * 0.8) ) #for rank approx
	d = matX.shape[1]

	matW = numpy.random.standard_normal((d,s)) / sigma
	vecV = 2 * numpy.pi * numpy.random.rand(1,s)

	matL = numpy.dot(matX, matW) + vecV 

	del matW

	matL = numpy.cos(matL) * numpy.sqrt(2. / s)
	matUL, vecSL, matVL = numpy.linalg.svd(matL, full_matrices = False)

	return matUL[:,0:k], vecSL[0:k]

'''
matX = numpy.random.randn(10,5)
sigma = 2
s = 3
print nystrom(matX, sigma, s)
#print RandomFeature(matX, sigma, s)
'''