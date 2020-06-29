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

    # TODO modify function to work with any number of features
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
    x = all Xs - the function iterates through all Xs in form [[1, x[0][1]], [1, x[1][1]], ..., [1, x[m-1][1]]]
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

def genData(dataSetSize=100, randomness=4.5, gradient=0.1, intercept=0):
    # all data
    dataSet = np.array([])

    LowerBound = gradient * np.arange(0, dataSetSize)
    
    for i in range(dataSetSize):
        if np.random.randint(0, 2) == 0:
            dataSet = np.append(dataSet, LowerBound[i]+(np.random.rand()*randomness)+intercept)
        else:
            dataSet = np.append(dataSet, LowerBound[i]-(np.random.rand()*randomness)+intercept)
    
    return dataSet

def genCluster(size=100, centre=[0, 0], maxDev=5):
    cluster = []
    for _ in range(size):
        if np.random.randint(0, 2) == 0:
            XSign = 1
        else:
            XSign = -1
        
        if np.random.randint(0, 2) == 0:
            YSign = 1
        else:
            YSign = -1
        # cluster.append([centre[0]+XSign*np.random.rand()*maxDev, centre[1]+YSign*np.random.rand()*maxDev])
        xValue = XSign*np.random.rand()*maxDev
        cluster.append([centre[0]+xValue, centre[1]+np.random.rand()*YSign*sqrt(maxDev**2-(xValue**2))])

    return cluster
