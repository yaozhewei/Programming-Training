#This is main part of sketeched regression method

import numpy

def rrSolver(matU, vecS, matV, vecUy, ngamma):
	d = vecUy.shape[0]
	vecSig = vecS + ngamma / vecS
	vecRR = vecUy / vecSig.reshape(d,1)
	vecRR = numpy.dot(matV.T, vecRR)

	return vecRR


def empericalRRonce(matX, vecW, sigma, vecGamma, sketchsizes,matU, vecS, matV):

#def empericalRRonce(matX, vecW, sigma, vecGamma, sketchsizes):
	#matU, vecS, matV = numpy.linalg.svd(matX, full_matrices = False)

	'''
	Y = XW + sigma \Rightarrow argmin || Y - XW || + gamma | W |
	sketch the problem
	'''
# set up the problem

	Repeat = 10; 

	n, d = matX.shape
	vecF = numpy.dot(matX, vecW)
	vecNoise = numpy.random.randn(n, 1) * sigma
	vecY = vecF + vecNoise
	vecXy = numpy.dot(matX.T, vecY)

	lenGamma = len(vecGamma)

	lenS = len(sketchsizes)

	# using the triditional way to get the optimal solution
	print 'Optimal solution of LSP'
	riskRR = numpy.zeros( (lenGamma,1) )
	objRR = numpy.zeros( (lenGamma,1) )
	vecUy = numpy.dot(matU.T,vecY)

	for i in range(lenGamma):
		gamma = vecGamma[i]
		vecRR = rrSolver(matU, vecS, matV, vecUy, n * gamma)

		vecXw = numpy.dot(matX, vecRR)

		#risk 
		err = numpy.linalg.norm(vecF - vecXw)
		riskRR[i] = err ** 2 / n

		#obj func value
		err = numpy.linalg.norm(vecY - vecXw)
		reg = numpy.linalg.norm(vecRR)

		del vecRR
		del vecXw

		objRR[i] = err ** 2 / n + gamma * ( reg ** 2)

	resultDict = {'sigma' : sigma,
				  'vecGamma' : vecGamma,
				  'sketchsizes' : sketchsizes,
				  'riskRR' : riskRR,
				  'objRR' : objRR}

	# solve RR by uniform sampling
	print 'Uniform sampling solution'

	riskUnif = numpy.zeros( (lenGamma, lenS) )
	objUnif = numpy.zeros( (lenGamma, lenS) )

	for j in range(0,lenS):
		s = sketchsizes[j]
		Errors = numpy.zeros( (lenGamma, Repeat) )
		#print  Errors
		objs = numpy.zeros( (lenGamma,Repeat) )
		#print  objs

		for rep in range(Repeat):
			idx = numpy.random.choice(n, s, replace = False)
			idx = numpy.unique(idx)
			matXsketch = matX[idx,:] * numpy.sqrt( n/s )
			vecYsketch = vecY[idx] * numpy.sqrt( n/s )
			matUsketch, vecSsketch, matVsketch = numpy.linalg.svd(matXsketch, full_matrices = False)
			vecUy = numpy.dot(matUsketch.T, vecYsketch)

			for i in range(lenGamma):
				gamma = vecGamma[i]
				vecRR = rrSolver(matUsketch, vecSsketch, matVsketch, vecUy, n * gamma)

				vecXw = numpy.dot(matX, vecRR)

				#risk 
				err = numpy.linalg.norm(vecF - vecXw)
				Errors[i, rep] = err ** 2 / n

				#obj func value
				err = numpy.linalg.norm(vecY - vecXw)
				reg = numpy.linalg.norm(vecRR)
				objs[i, rep] = (err ** 2) / n + gamma * (reg ** 2)

				del vecRR
				del vecXw

			del vecUy
			del matXsketch
			del vecYsketch
			del matUsketch
			del vecSsketch
			del matVsketch

		riskUnif[:, j] = numpy.mean(Errors, axis=1)
		#print numpy.mean(objs, axis=1)
		objUnif[:, j] = numpy.mean(objs, axis=1)

	resultDict['riskUnif'] = riskUnif
	resultDict['objUnif'] = objUnif

	return resultDict



def empericalRR(matX, vecW, sigma, vecGamma, sketchsizes):

	matU, vecS, matV = numpy.linalg.svd(matX, full_matrices = False)

	Repeat = 10

	n, d = matX.shape
	lenGamma = len(vecGamma)
	lenS = len(sketchsizes)

	riskRR = numpy.zeros( (lenGamma,1) )
	objRR = numpy.zeros( (lenGamma,1) )

	riskUnif = numpy.zeros( (lenGamma, lenS) )
	objUnif = numpy.zeros( (lenGamma, lenS) )

	for rep in range(Repeat):
		print 'pass #' + str(rep)

		resultDict1 = empericalRRonce(matX, vecW, sigma, vecGamma, sketchsizes,matU, vecS, matV)

		obj = resultDict1['objRR']

		riskRR += resultDict1['riskRR'] / Repeat

		riskUnif  += resultDict1['riskUnif'] / Repeat
		#objUnif += resultDict1['objUnif'] / obj  /Repeat
		objUnif += resultDict1['objUnif']  /Repeat


	resultDict = {'sigma' : sigma,
				  'vecGamma' : vecGamma,
				  'sketchsizes' : sketchsizes,
				  'riskRR' : riskRR,
				  'riskUnif' : riskUnif,
				  'objUnif' : objUnif}
	return resultDict








 