import numpy
import sys
sys.path.append('../main/regression')
import syndata
import scipy.io 
import scipy.linalg
'''
n = 10000
d = 50
sigma =3

dataclass = 'NB'

matX, vecY, vecW = syndata.generate_LSR(n, d, dataclass, sigma)

#print(matX.shape)

dataDict = {'n' : n,
			'd' : d,
			'matX' : matX,
			'vecY' : vecY,
			'vecW' : vecW }

outputFileName = dataclass + '_n=' + str(n) + '_d=' + str(d) + '_sigma=' + str(sigma) + '.mat'

scipy.io.savemat(outputFileName, dataDict)

mathDict = scipy.io.loadmat(outputFileName)
'''

n = 10000
d = 50

dataclass = 'NB'

matX, vecW = syndata.generate_XW(n, d, dataclass)

#print(matX.shape)

dataDict = {'n' : n,
			'd' : d,
			'matX' : matX,
			'vecW' : vecW }

outputFileName = dataclass + '_n=' + str(n) + '_d=' + str(d) + '.mat'

scipy.io.savemat(outputFileName, dataDict)

#mathDict = scipy.io.loadmat(outputFileName)





