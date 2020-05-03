import numpy as np
from math import sqrt

def meanOf(data):
    mean = 0
    for i in data:
        mean += i
    mean = mean/len(data)
    return mean

def variance(data):
    mean = meanOf(data)
    variance = 0
    for i in data:
        variance += (i-mean)**2
    variance = variance/len(data)
    return variance

def sampleVariance(data):
    sampleVariance = (variance(data)*len(data))/(len(data)-1)
    return sampleVariance

def standardDev(data):
    standardDev = sqrt(variance(data))
    return standardDev

def sampleStandardDev(data):
    sampleStandardDev = sqrt(sampleVariance(data))
    return sampleStandardDev

def MAD(data):
    mean = meanOf(data)
    MAD = 0
    for i in data:
        MAD += abs(mean-i)
    MAD = MAD/len(data)
    return MAD

def cor(xdata, ydata):
    # Pearson's corrlation coefficient
    cor = 0
    xmean = meanOf(xdata)
    ymean = meanOf(ydata)
    try:
        num = np.sum(np.full([len(xdata), ], (xdata-xmean)*(ydata-ymean)))
        denum = sqrt(np.sum(((xdata-xmean)**2)*np.sum((ydata-ymean)**2)))
    except IndexError:
        print("The iterables were not of the same size")
    cor = num/denum
    return cor

def gradientDescentLinearRegression(Xs, Ys, alpha, iterations):
    # GENERATING COST FUNCTION WILL PRODUCE SEEMINGLY NON-CONVEX CURVE BUT IT IS ACTUALLY CONVEX
    theta = [0, 0]
    Xs = [[1, x] for x in Xs]

    '''
    theta = [theta0, theta1] - just parameters
    xi = [1, x] - 1 is there to multiply with theta0 - xi is ONE training example
    '''
    # Does numpy automatically tranpose arrays when finding dot product?
    h = lambda theta, xi: np.dot(theta, xi)
    
    '''
    theta = [theta0, theta1]
    Xs = all Xs - the function iterates through all Xs in form [[1, x0], [1, x1], ..., [1, xn]]
    Ys = all Ys - the fucntion iterates through all Ys
    '''
    J = lambda theta, Xs, Ys: 1/(len(Ys))*np.sum([np.power((h(theta, Xs[i]) - Ys[i]), 2) for i in range(len(Ys))])
    
    '''
    theta = [theta0, theta1]
    x = all Xs - the function iterates through all Xs in form [[1, x0], [1, x1], ..., [1, xn]]
    y = all Ys - the fucntion iterates through all Ys
    j = for the x[i][j] part - needs to be provided by the calling function
    process: when updating jth theta pass the same j to dJ
    '''
    dJ = lambda theta, x, y, j: (1/(len(Ys))) * np.sum( [ (h(theta, x[i]) - y[i])*x[i][j] for i in range(len(y)) ] )

    for _ in range(iterations):
        tmpTheta = theta.copy()
        for j in range(len(theta)):
            theta[j] = tmpTheta[j] - alpha*dJ(tmpTheta, Xs, Ys, j)

    return theta

def linReg(xdata, ydata, sample: bool):
    # linear regression using Pearson's correlation coefficient
    m = 0
    r = cor(xdata, ydata)
    if sample == False:
        m = r*(standardDev(ydata)/standardDev(xdata))
    else:
        m = r*(sampleStandardDev(ydata)/sampleStandardDev(xdata))

    c = meanOf(ydata)-(m*meanOf(xdata))

    return [m, c]
    
def gApprox(xdata, ydata):
    m = 0
    current = 0
    for i in range(len(xdata)):
        if i != len(xdata)-1 and xdata[i] != xdata[i+1]:
            m += (ydata[i+1]-ydata[i])/(xdata[i+1]-xdata[i])
            current = i
        elif i != len(xdata)-1 and xdata[i] == xdata[i+1]:
            m += (ydata[i+1]-ydata[current])/(xdata[i+1]-xdata[current])

    m = m/len(xdata)
    c = meanOf(ydata)-(m*meanOf(xdata))
    
    return [m, c]

def MADReg(xdata, ydata):
    r = cor(xdata, ydata)
    m = r*(MAD(ydata)/MAD(xdata))
    c = meanOf(ydata)-(m*meanOf(xdata))
    return [m, c]

def genData(dataSetSize=100, randomness=4.5, UpperBound=20):
    # genrates random data that has size dataSetSize, a rather arbitrary randomness value of randomness and the UpperBound of the data is 20
    # however the values are still likely to go over 20 after a random number is added to them
    dataSet = np.array([])

    # y = range*x
    LowerBound = UpperBound/dataSetSize * np.arange(0, dataSetSize)
    for i in range(dataSetSize):
        if np.random.randint(0, 2) == 0:
            dataSet = np.append(dataSet, LowerBound[i]+(np.random.rand()*randomness))
        else:
            dataSet = np.append(dataSet, LowerBound[i]-(np.random.rand()*randomness))
    
    return dataSet

def prepData(data1, data2):
    # orders data1 and data2 such that data1 is ordered in ascending order and each value in data2 still corresponds to it's initial 
    # value in data1
    # eg: data1 = [3, 1, 2]
    #     data2 = [50, 70, 40]
    # result will be: 
    # data1 = [1, 2, 3]
    # data2 = [70, 40, 50]
    allData = []
    for i in range(len(data1)):
        allData.append([data1[i], data2[i]])
    allData.sort()

    for i in range(len(allData)):
        data1[i] = allData[i][0]
        data2[i] = allData[i][1]
    
    return data1, data2
