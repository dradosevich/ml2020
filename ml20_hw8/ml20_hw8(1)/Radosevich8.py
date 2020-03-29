
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpl
import os
import random
import datetime
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

# this is the COMPAS data set (recidivism prediction)
data = pd.read_csv("compas.csv")

# drop duplicate rows
data.drop_duplicates(keep = False, inplace = True)
data = np.asarray(data)
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

# let us use 10% for training to reduce computational time
# but keep in mind kernel methods can easily overfit
n_train = int(n*0.1)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

#Danny's Variable Declarations

#lists for plotting
krrTrainError=[]
krrTestError=[]

akrrTrainError=[]
akrrTestError=[]

krrTime = []
akrrTime = []
#const lambda and variable sigmas
lam = .01
sigma =[.001,.01,.1,1,10,100,1000]

#helper function Declarations

#function to get the alpha term
def dateBreak(time):
    time = str(time).split(" ")
    return time[1]
def getAlph(k,x,y):
    k = np.asarray(k)
    alpha = k+lam*(np.identity(np.size(x[:,1])))
    alpha = np.linalg.inv(alpha) #step one the inversion
    alpha = alpha.dot(y)
    #print(alpha.shape)
    return alpha
#function to get the gaussian result
def gauss(a,b,sig):
    return np.exp(-(np.square((np.linalg.norm(a-b)))/(np.square(2*sig))))
#function to get K
def getk(x, sig):

    # here, evaluate testing MSE and training MSE
    k = np.zeros(shape=(x.shape[0],x.shape[0])) #create our k
    X = np.asarray(x) #transform sample train to a np array
    i = 0 #bound
    for xi in X:
        #if i%100 ==0:
            #print(i)
        j = 0
        for xj in X:
            #print(j)
            k[i,j] = gauss(xi,xj,sig)
            #print(k[i,j])
            j+=1
        i+=1
    return k
def getTestK(x,z,sig):
    #x is sample train
    #z is sample_test
    zx,zy = z.shape
    xx,xy = x.shape
    k = np.zeros(shape=(zx,xx))
    x = np.asarray(x)
    i = 0
    for zi in z:
        #if i%1000 ==0:
            #print(i)
        j = 0
        for xi in x:
            k[i,j] = gauss(zi,xi,sig)
            j+=1
        i+=1
    return k

# now, build your KRR, AKRR and (if you choose to) AKRR+

#function for kernel ridghe regression

def krr(k,testk,x,y):
    #k is the K matrix
    #x is the training data
    #y is the label
    #first get alpha
    alpha = getAlph(k,x,y)
    print(k.shape,alpha.shape,label_test.shape)
    krr_mse_train = np.mean(np.square(k.dot(alpha)-y))
    krr_mse_test = np.mean(np.square(testk.dot(alpha)-label_test))
    print("krr mse train", krr_mse_train)
    print("krr mse test", krr_mse_test)
    krrTrainError.append(krr_mse_train)
    krrTestError.append(krr_mse_test)

#function for the akrr
def akrr(k,ktest,x,y):
    #print(k)
    #k is the K matrix
    #x is the training data
    #y is the label
    #things
    #first get alpha
    alpha = getAlph(k,x,y)

    #akrr_mse_train = np.square(np.linalg.norm(k.dot(alpha)-y))
    akrr_mse_train = np.mean(np.square(k.dot(alpha)-y))
    #print(ktest.shape,alpha.shape,label_test.shape)
    akrr_mse_test = np.mean(np.square(ktest.dot(alpha)-label_test))
    print("akrr mse train", akrr_mse_train)
    print("akrr mse test", akrr_mse_test)
    akrrTrainError.append(akrr_mse_train)
    akrrTestError.append(akrr_mse_test)
#main function
def main():
    akin = [] #vector for the modified k
    nn,pp =sample_train.shape
    while(len(akin)<120):
        temp = random.randint(0,nn-1)
        if temp not in akin:
            akin.append(temp)
    akin.sort()

    #declare the new training and label
    ktrain = [] #akrr trining data
    klabel = [] #akrr label data

    for ind in akin: #go through all the
        ktrain.append(sample_train[ind])
        klabel.append(label_train[ind])
    print(type(ktrain))
    ktrain = np.asarray(ktrain)
    klabel = np.asarray(klabel)
    print(klabel.shape)
    # here, evaluate testing MSE and training MSE
    #loop to go through for the varying sigma
    for sig in sigma: #go thorugh my 7 sigma choices
        #calculate K
        print("Currently on sigma ",sig)
        os.system("date")
        rowSize, colSize = ktrain.shape #store the shape of sample_train
        #print("sample is a %d by %d" %(rowSize,colSize))
        start = datetime.datetime.now()
        krr(getk(sample_train,sig),getTestK(sample_train,sample_test,sig),sample_train,label_train)
        end = datetime.datetime.now()
        krrTime.append((start-end).total_seconds())
        start = datetime.datetime.now()
        akrr(getk(ktrain,sig),getTestK(ktrain,sample_test,sig),ktrain,klabel)
        end = datetime.datetime.now()
        krrTime.append((start-end).total_seconds())
    print(np.mean(krrTime))
    print(np.mean(akrrTime))
    os.system("date")
    mpl.plot(sigma,krrTrainError,label="krr training")
    mpl.plot(sigma,krrTestError,label="krr testing")
    mpl.ylabel("Error")
    mpl.xlabel("Sigma")
    mpl.title("Error with respect to sigma")
    mpl.legend()
    mpl.show()
    mpl.clf()
    mpl.plot(sigma,akrrTrainError,label="akrr training")
    mpl.plot(sigma,akrrTestError,label="akrr testing")
    mpl.ylabel("Error")
    mpl.xlabel("Sigma")
    mpl.title("Error with respect to sigma")
    mpl.legend()
    mpl.show()

#call main
main()
