import numpy

def generate_Cov_Matrix(d):
	# This is to generate the covriance matrix C
	C = numpy.zeros( (d,d) )

	for i in range(0,d):
		for j in range(0,d):
			C[i,j] = 2 * 0.05 ** (abs(i-j))
	return C


# test: print generate_Cov_Matrix(3)

def generate_rnd_of_T_Dist(mu, Cov, v, n):
	# Generate a rand matrix with t-distribution n * d using the formula X = Y / sqrt (Z), where Y is a standard gaussian dist and Z is possion dist.
	d = len(Cov);
	Z = numpy.tile(numpy.random.gamma(v/2., 2./v, n), (d,1)).T

	Y = numpy.random.multivariate_normal(numpy.zeros(d), Cov, n)

	return mu + Y / numpy.sqrt(Z)



def generate_X_Matrix(n,d,dataclass):
	# Generate X of Y = X * beta + eplison
	v = 2; #using the same setting in review paper, Ping Ma used 3

	mu = numpy.ones(d);
	cov_Matrix = generate_Cov_Matrix(d);

	# generate matrix U
	if dataclass == 'UG' or dataclass == 'UB':
		matA = numpy.random.multivariate_normal(mu, cov_Matrix, n)
	elif dataclass == 'NG' or dataclass == 'NB':
		matA = generate_rnd_of_T_Dist(mu, cov_Matrix, v, n)

	#print numpy.rank(matA)
	matU = numpy.linalg.qr(matA, mode='reduced')[0]

	# generate matrix S = diag( sigma )
	if dataclass == 'NG' or dataclass == 'UG':
		vecSigma = numpy.linspace(1, 0.1, d)

	elif dataclass == 'NB' or dataclass == 'UB':
		vecSigma = numpy.logspace(0, -6, d)

	#generate matrix V
	matV = numpy.random.randn(d, d)

	matV = numpy.linalg.qr(matV, mode='reduced')[0] 

	matX = matU * vecSigma.reshape(1,d)
	matX = numpy.dot(matX,matV)

	return matX

# test: print generate_X_Matrix(3,2,'NG')

def generate_XW(n, d, dataclass):
	d1 = int(numpy.ceil(d / 5));

	vecW1 = numpy.ones( (d1,1) );
	vecW2 = numpy.ones( (d - 2 * d1, 1) );
	vecW = numpy.concatenate( (vecW1, .1 * vecW2, vecW1) )

	matX = generate_X_Matrix(n, d, dataclass)
	return matX, vecW


def generate_LSR(n, d, dataclass, sigma):

	d1 = int(numpy.ceil(d / 5));

	vecW1 = numpy.ones( (d1,1) );
	vecW2 = numpy.ones( (d - 2 * d1, 1) );
	vecW = numpy.concatenate( (vecW1, .1 * vecW2, vecW1) )

	matX = generate_X_Matrix(n, d, dataclass)

	vecY = numpy.dot(matX, vecW) + numpy.random.randn(n,1) * sigma;

	return matX, vecY, vecW

#test: print generate_LSR(20,10,'NB',1)






