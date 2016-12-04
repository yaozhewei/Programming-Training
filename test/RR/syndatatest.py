import numpy
import sys
sys.path.append('../main/regression')
import regression
import scipy.io 

matfn = '/Users/Wemyss/Regression/data/NB_n=10000_d=50.mat'
mathDict = scipy.io.loadmat(matfn)

#print mathDict['matX'].shape

matX = mathDict['matX']

vecW = mathDict['vecW']

#resultDict = regression.empericalRRonce(matX, vecW, 0, [0,1], [3, 5, 10,50])
resultDict = regression.empericalRR(matX, vecW, 0, [0,1], [3, 5, 10,50])


outputFileName = 'result' + '.mat'

scipy.io.savemat(outputFileName, resultDict)
