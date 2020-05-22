
#Danny Radosevich
#Wroked with Mason Jonson 
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
# this is the COMPAS data set (recidivism prediction)
data = pd.read_csv("compas.csv")

# drop duplicate rows
data.drop_duplicates(keep = False, inplace = True)
data = np.asarray(data)
data = np.insert(data,0,1,1)
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

# choose the percentage of classification methods yourself
n_train = int(n*.2)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

lam = .75
predsvec = []
# now, implement logistic regression
# you may want to implement both optimization methods inside the same loop to save time
# remember to store your training errors of both methods after one update

def prob(y,x,beta):

    toreturn = np.exp(np.dot(np.transpose(x),beta))
    #toreturn = 1/(1+toreturn)
    """
    if toreturn < 0:
        toreturn = 1-(1-1/(1+math.exp(toreturn)))
    else:
        toreturn = 1-(1/(1+math.exp(-toreturn)))
    #return toreturn
    """
    toreturn = 1/(1+toreturn)
    if (y == 0):
        #prob when y = 0
        return 1-toreturn
    elif (y == 1):
        #prob when y = 1
        return toreturn
def predTwo(x,beta):
    tmp = beta[0]
    for i in range (len(x)-1):
        tmp += beta[i+1]*x[i]
    return 1.0/(1.0-math.exp(-tmp))

def pvec(beta):
    pv = np.zeros(len(sample_train))
    for i in range(len(sample_train)):
        probr = prob(1,sample_train[i],beta)
        #print(type(probr))
        if isinstance(probr,np.float64):
            #print("------")
            pv[i] = probr
        else:
            pv[i] = 0

    #print(pv)
    return np.transpose(pv)
def getW(beta):
    x = sample_train
    w = np.zeros((len(x),len(x)))
    for i in range(len(x)):
        w[i][i]= np.dot(prob(1,x[i],beta),prob(0,x[i],beta))
    return w

def gPrime(beta):
    gpvec = pvec(beta)
    #print(p)
    return np.dot(np.transpose(sample_train),(label_train-gpvec))
def gDoublePrime(beta):
    ret = np.dot(np.transpose(sample_train),getW(beta))
    return np.dot(ret,sample_train)
def checkDelt(delt):
    for el in delt:
        #print(el)
        if el > .000001:
            return False
    return True
def doNewt(times):
    cont = True
    beta = np.zeros(p) #inital beta

    ident = np.identity(p) #make an identity matrix
    ident[0][0] = 0 # set the first elem to be zero


    for i in range(times):
        gp = gPrime(beta) #get g'
        gpp = gDoublePrime(beta) #get g''
        #print("gpp shape",gpp.shape)
        #print("-----------")
        #print(gpp)
        inv = np.linalg.inv(gpp) #get the inversion of gpp
        delt = -np.dot(inv,gp) #calculate the delta of beta
        print("delta\n",delt)

        #update beta
        #beta = beta+delt
        pone = np.add(gpp, 2*lam*ident)
        pone = np.transpose(pone)
        ptwo = 2*lam*ident
        ptwo = np.dot(ptwo,beta)
        ptwo = np.add(inv,ptwo)
        beta = beta + np.dot(pone,ptwo)
        #print("----------------------")
    return beta
def doGD(times):
        beta = np.zeros(p)
        for i in range(p):
            #print(i)
            beta[i] = .0001

        for i in range(times):
            terr = 0
            for j in range(len(sample_train)):
                pred = 0
                for k in range(len(sample_train[j])):
                    pred += beta[k]*sample_train[j][k]
                if pred < 0:
                    pred = 1 - 1/(1+math.exp(pred))
                else:
                    pred = 1.0/(1.0+math.exp(-pred))
                error = label_train[j] - pred
                terr += error**2
                for k in range(p):
                    beta[k] = beta[k] + lam*error*pred*(1.0-pred)*sample_train[j][k]
        return beta
        #while not checkDelt(prevX-curX):

tries = [1,5,10,20,50,100,1000]

newerr = []
gderr = []

for t in tries:
    print(t)
    #newd = doNewt(t)
    #print(doGD(t))
    #newB = doNewt(t)
    bet = doGD(t)
    #print(newB)
    #print(bet)
    myp = pvec(bet)
    #print(type(myp))
    for i in range(len(myp)):
        myp[i] = np.round(myp[i])
    #myp = np.round(p)
    total = 0
    misses = 0
    tmiss = 0
    print(myp.shape, label_train.shape,label_test.shape)
    for i in range(len(myp)):
        total = i
        if(myp[i]!=label_train[i]):
            misses+=1
        if(myp[i]!=label_test[i]):
            tmiss+=1
    gderr.append(misses/total)
    print("GD Error",misses/total)
    print("GD TEST",tmiss/total)
    #print(myp)
plt.plot(gderr, label ="GD error" )
plt.title("Error vs iterations")
plt.legend()
x = np.array([0,1,2,3,4,5,6])
plt.xticks(x,tries)
plt.show()
# now plot the two convergence curves

# ......

# ......
