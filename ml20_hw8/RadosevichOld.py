
import numpy as np
import matplotlib.pyplot as mpl
import os
import random
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
#import timeit

# this is the COMPAS data set (recidivism prediction)
data = np.loadtxt('compas.csv', delimiter=',')

sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

lam = .01
sigma =[.001,.01,.1,1,10,100,1000]
# a standard split is 75% for training and 25% for testing
# previously, we use 1% for training in order to show overfitting
n_train = int(n*0.75)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

krrTrainError=[]
krrTestError=[]

akrrTrainError=[]
akrrTestError=[]

# now, build your KRR, AKRR and (if you choose to) AKRR+
'''
def findbeta(k,a,y):
    #j(a)=(Ka-Y)^t(Ka-Y)+lama^tKa
    temp = (k.dot(a))
    temp = temp-y
    beta = np.transpose(beta)
    beta = beta.dot(temp)
    beta = beta + lam*(np.transpose(a).dot(k).dot(a))
    return beta
'''
def getAlph(k,x,y):
    ident = np.zeros((y.shape[0],y.shape[0]))
    k = np.asarray(k)
    alpha = k+lam*(np.identity(np.size(x[:,1])))
    #print(alpha)
    alpha = np.linalg.inv(alpha) #step one the inversion
    alpha = alpha.dot(y)
    print(alpha.shape)
    return alpha

def krr(k,x,y):
    #k is the K matrix
    #x is the training data
    #y is the label
    #first get alpha
    alpha = getAlph(k,x,y)

    krr_mse_train = np.square(np.linalg.norm(k.dot(alpha)-y))
    krr_mse_test = np.mean(np.square(k.dot(alpha)-labeltest))
    krrTrainError.append(krr_mse_train)
    krrTestError.append(krr_mse_test)
def akrr(k,x,y):
    print(k)
    #k is the K matrix
    #x is the training data
    #y is the label
    #things
    #first get alpha
    alpha = getAlph(k,x,y)

    #akrr_mse_train = np.square(np.linalg.norm(k.dot(alpha)-y))
    akrr_mse_train = np.mean(np.square(k.dot(alpha)-y))
    print("mse", akrr_mse_train)
    #akrr_mse_test = np.mean(np.square(k.dot(beta)-label_test))
    #akrrTrainError.append(akrr_mse_train)
    #akrrTestError.append(akrr_mse_test)

def gauss(a,b,sig):
    return np.exp(-(np.square((np.linalg.norm(a-b)))/(np.square(2*sig))))
def getk(x, sig):
# here, evaluate testing MSE and training MSE
    k = np.zeros(shape=(x.shape[0],x.shape[0])) #create our k
    X = np.asarray(x) #transform sample train to a np array
    i = 0 #bound
    for xi in X:
        if i%1000 ==0:
            print(i)
        j = 0
        for xj in X:
            #print(j)
            k[i,j] = gauss(xi,xj,sig)
            #print(k[i,j])
            j+=1
        i+=1
    return k

signum = 0
#for sig in sigma: #go thorugh my 7 sigma choices
    #calculate

#    os.system("date")
    #rowSize, colSize = sample_train.shape #store the shape of sample_train
#    krr(getk(sample_train,sig),sample_train,label_train)
    #signum+=1

akin = []
while(len(akin)<1200):
    temp = random.randint(0,12000)
    if temp not in akin:
        akin.append(temp)
akin.sort()

ktrain = []
klabel = []

for ind in akin:
    ktrain.append(sample_train[ind])
    klabel.append(label_train[ind])
ktrain = np.asarray(ktrain)
klabel = np.asarray(ktrain)

for sig in sigma: #go thorugh my 7 sigma choices
    #calculate K
    print("Currently on sigma ",sig)
    os.system("date")
    rowSize, colSize = ktrain.shape #store the shape of sample_train
    #print("sample is a %d by %d" %(rowSize,colSize))
    akrr(getk(ktrain,sig),ktrain,klabel)
    signum+=1

#mpl.plot(sigma,krrTrainError,label="krr training")
#mpl.plot(sigma,krrTestError,label="krr testing")
#mpl.ylabel("Error")
#mpl.xlabel("Sigma")
#mpl.title("Error with respect to sigma")
#mpl.legend().show()
#mpl.plot(sigma,akrrTrainError,label="akrr training")
#mpl.plot(sigma,akrrTestError,label="akrr testing")
#mpl.ylabel("Error")
#mpl.xlabel("Sigma")
#mpl.title("Error with respect to sigma")
#mpl.legend().show()
