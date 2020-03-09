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

def linReg(xdata, ydata, sample: bool):
    # linear regression
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
