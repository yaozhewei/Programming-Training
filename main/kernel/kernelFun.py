import numpy

def rbfkernel(matX1, matX2, sigma):
	n1 = matX1.shape[0]
	n2 = matX2.shape[0]

	rowNormX1 = numpy.sum( numpy.square(matX1), 1 ) / 2.
	rowNormX2 = numpy.sum( numpy.square(matX2), 1 ) / 2.

	K = numpy.dot(matX1, matX2.T) - rowNormX1.reshape(n1, 1) - rowNormX2.reshape(1, n2)

	K = K / (sigma ** 2)

	return numpy.exp(K)
'''
matX1 = numpy.random.randn(5,3);
matX2 = numpy.random.randn(3,3);
sigma = 2;

print rbfkernel(matX1,matX2,sigma)
'''


