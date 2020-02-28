#Danny Radosevich
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


randLam = np.random.uniform(low=0, high=100,size=(1000)) #array of 30 random lambdas
randLam.sort()#sort it
# create an array of weights w, and then preset its values
# for example, the weight of x_i is w_i = 1.
weights = np.zeros((n_train,n_train))#assigned
for i in range(n_train):
    for j in range(n_train):
        if i==j and i != 0 :
            weights[i,j] = 1
            #ident[i,j] = 1

ident = np.zeros((p,p))
for i in range(p):
    for j in range(p):
        if i==j and i != 0:
            #weights[i,j] = 1
            ident[i,j] = 1
trainError=[]
testError=[]
#beta=(xTx+lamI0)^(-1)xTy
def fig1():
    beta=np.zeros((p+1))
    #for i in range(n):
    #beta=(np.linalg.inv(((data.transpose()).dot(data))+(.01*ident[i])).dot(data.transpose().dot(label)))
    for lam in randLam:
        xTrans = np.transpose(sample_train)#save xTranspose
        xTw=xTrans.dot(weights)#mult trans by w
        xTx = xTw.dot(sample_train) #mult xTw by x
        lamI = lam*ident #get our I by our scalar
        innerIn = np.linalg.inv(xTx+lamI)#get the inversion
        innerNext = innerIn.dot(xTrans).dot(weights) #mult the inversion by the trans then weight
        beta = innerNext.dot(label_train) #finally multiply by y to get beta

        mse = np.mean((np.square(sample_train.dot(beta)-label_train)))#calc MSE
        mseTest = np.mean((np.square(sample_test.dot(beta)-label_test)))#calc MSE

        trainError.append(mse)#put it into the matrix for plotting
        testError.append(mseTest)

    mpl.plot(randLam, trainError, label="training")
    mpl.plot(randLam,testError, label="testing")

    mpl.ylabel("Error")
    mpl.xlabel("lambda")
    mpl.title("Average over MSE 1000 Lamdas with Lambda range of 0-100")
    mpl.legend()
    mpl.show()


def fig2():
    toPlotMin =[]
    toPlotMaj =[]
    weightMin =[]
    weightMaj =[]
    newWeight = np.zeros((n_train,n_train))
    minWeight = [.001,.01,.1,1,10,100,1000]
    minority = np.where(sample_train[0]==1) #where there are minoritiesclear
    majority = np.where(sample_train[0]==0) #where there are majorities

    for w in minWeight:
        for i in range(n_train):
            for j in range(n_train):
                if i==j and i != 0 and i in minority:
                    newWeight[i,j] = w
                elif i==0:
                    newWeight[i,j]=0
                elif i==j:
                    newWeight[i,j]=1
        xTrans = np.transpose(sample_train)#save xTranspose
        xTw=xTrans.dot(newWeight)#mult trans by w
        xTx = xTw.dot(sample_train) #mult xTw by x
        lamI = .01*ident #get our I by our scalar
        innerIn = np.linalg.inv(xTx+lamI)#get the inversion
        innerNext = innerIn.dot(xTrans).dot(weights) #mult the inversion by the trans then weight
        beta = innerNext.dot(label_train) #finally multiply by y to get beta
        pred = sample_test.dot(beta)#create predicitons
        #print(pred)
        sqer = np.square(pred-label_test) #create square error
        #print(sqer)
        for iter in range(len(sqer)):
            if sample_test[iter,0]==1:
                weightMin.append(sqer[iter])
            else:
                weightMaj.append(sqer[iter])
        print("min, ",len(weightMin))
        print("maj ",len(weightMaj))
        toSumMin = np.array(weightMin)
        toSumMaj = np.array(weightMaj)
        mse = (np.sum(toSumMin)/len(toSumMin))
        print("min", mse)
        toPlotMin.append(mse)
        mse2 = np.sum(toSumMaj)/len(toSumMaj)
        print("maj",mse2)
        toPlotMaj.append(mse2)
        weightMin=[]
        weightMaj=[]


    mpl.plot( toPlotMin, label="Minority")
    mpl.plot(toPlotMaj, label="Majority")

    mpl.ylabel("Error")
    mpl.xlabel("weight")
    mpl.title("differing weights for minority communities")
    mpl.legend()
    mpl.show()



fig2()
