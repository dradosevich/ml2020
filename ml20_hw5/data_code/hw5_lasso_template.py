
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

lam =.01

historicalTest=[]
historicalTrain=[]

# implement lasso
# you should record training & testing MSEs after every update (for Figure 1 plotting)
# you should also record the number of non-zero elements in beta after every update
def lasso():
    for i in range(p):
        myrand = np.random.randint(0,p)
        result =0
        fakebeta = beta
        fakebeta[0] =0
        if myrand==0:
            for k in range(n_train):
                xtrans = np.transpose(sample_train[k,:])
                xtb = xtrans.dot(fakebeta[1:])
                beta[myrand] = -(1/n)*(xtb-label_train[k])
        else:
            xj=sample_train[:,myrand]
            xtran = np.transpose(xj)
            tempBeta = np.delete(fakebeta,myrand)
            ajprime = sample_train.dot(tempBeta)
            aj = ajprime-label_train
            xta = 2*xtran.dot(aj)
            topOne = -lam-xta
            bottom = 2*xj.dot(np.transpose(xj))
            topTwo = lam-xta
            if xta < -lam:
                beta[myrand] = (topOne/bottom)
            elif xta > lam:
                beta[myrand] = (topTwo/bottom)
            else:
                beta[myrand] = 0

# Figure 1: plot your historical training MSE and testing MSE into two curves
def fig1():
    mpl.plot( historicalTrain, label="training")
    mpl.plot(historicalTest, label="testing")

    mpl.ylabel("Error")
    mpl.xlabel("")
    mpl.title("Figure 1")
    mpl.legend()
    mpl.show()

# Figure 2: plot your number of non-zero elements in beta
def fig2():
    print()

# For Figure 3, you will need to run multiple times with different lambda's and plot the converged results
def fig3():
    print()

# For Figure 4, you will need to run multiple times with different lambda's and plot the converged results
def fig4():
    print()
lasso()
#print(beta)
#fig2()
