
import numpy as np
import matplotlib.pyplot as mpl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

data = np.loadtxt('crimerate.csv', delimiter=',')

sample = data[:,0:-1]
label = data[:,-1]

[n,p] = sample.shape

n_train = int(n*0.01)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

# randomly initialize beta
beta = np.random.uniform(low=-1,high=1,size=(p+1))



historicalTest=[]
historicalTrain=[]
betaCount = []

# implement lasso
# you should record training & testing MSEs after every update (for Figure 1 plotting)
# you should also record the number of non-zero elements in beta after every update
def lasso(lam):
    caseOne = 0
    caseTwo = 0
    caseThree = 0
    for i in range(1000):
        count = 0
        for d in range(len(beta)):
            if beta[d] !=0:
                count +=1
        betaCount.append(count)
        #print(count)
        myrand = np.random.randint(0,p)
        result = 0
        fakebeta = beta
        #fakebeta[0] = 0
        if myrand==0:
            for k in range(n_train):
                xtrans = np.transpose(sample_train[k,:])
                xtb = xtrans.dot(fakebeta[1:])
                beta[myrand] = -(1/n_train)*(xtb-label_train[k])
        else:
            xj=sample_train[:,myrand] #create x_j
            #print(xj)
            xtran = np.transpose(xj) #create the transpose of x_j
            tempBeta = np.delete(fakebeta,myrand) #remove the element to align sizes
            #print(tempBeta)
            ajprime = sample_train.dot(tempBeta) #create aj prime to split up operations
            aj = ajprime-label_train # make aj= X*B[-j] - Y
            #print(aj)
            xtTwo = 2*xtran # 2* the transpose of x
            xta = xtTwo.dot(aj) # crate our check for cases
            topOne = np.negative(lam)-xta # create first nom for assignment
            #the above is -lamda - xta
            bottom = 2*(xj.dot(xtran)) #create the denom 2*||x_j||^2
            topTwo = lam-xta#create second nom, lam -xta
            print("------------------------------------------")
            print(xta)
            #print(beta)
            if lam < np.negative(xta):
                beta[myrand] = (topOne/bottom)
                print(topOne,bottom)
                caseOne += 1
            elif lam < xta:
                caseTwo += 1
                beta[myrand] = (topTwo/bottom)
                print(topOne,bottom)
            else:
                print("in three")
                caseThree += 1
                beta[myrand] = 0
            xta =0
            print("------------------------------------------")
    print(caseOne,caseTwo,caseThree)
# Figure 1: plot your historical training MSE and testing MSE into two curves
def fig1():
    mpl.plot(historicalTrain, label="training")
    mpl.plot(historicalTest, label="testing")

    mpl.ylabel("Error")
    mpl.xlabel("")
    mpl.title("Figure 1")
    mpl.legend()
    mpl.show()

# Figure 2: plot your number of non-zero elements in beta
def fig2():
    mpl.plot(betaCount)
    mpl.show()
    print()

# For Figure 3, you will need to run multiple times with different lambda's and plot the converged results
def fig3():
    print()

# For Figure 4, you will need to run multiple times with different lambda's and plot the converged results
def fig4():
    print()

lam =.01
lasso(lam)

print(beta)
#fig2()
randLam = np.random.uniform(low=0, high=100,size=(1000))
