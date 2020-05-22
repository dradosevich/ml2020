
# you cannot use PCA library
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lr #for the section where libraries are allowed
from sklearn.metrics import mean_squared_error

#
data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape


# in this task, we will consider a prediction problem
# let's first split data into training and testing sets
sample_train = sample[0:int(0.75*n),:]
label_train = label[0:int(0.75*n)]
sample_test = sample[int(0.75*n):,:]
label_test = label[int(0.75*n):]


# let's learn PCA projection vectors from the training sample (i.e., "sample_train")

def getMew():
    n = 0
    for x in sample_train:
        n +=1
    #print(mew)
    #mew = np.asarray(mew)
    return (sample_train.dot(1/n))
def getSigma():
    #stuff und things
    mew = getMew()
    pca = []
    n = 0
    line =""
    for x in sample_train:
        n += 1
    xmew = sample_train-mew
    trans = np.transpose(xmew)
    sig = xmew.dot(trans)
    sig = np.asarray(sig)
    return (sig.dot(1/n))


# store your k projection vectors in a matrix "w_train" (similar to w in [2])
# note: here you need to vary k to get different prediction mse

sig = getSigma()
evals, evec = np.linalg.eig(sig)
#w_train = np.transpose(sig).dot(sample_train)
w_train = evec
xtild = np.transpose(sig).dot(sample_train)
print(xtild.shape)



# next, project both training sample and testing sample onto w_train
# both training and testing instances will now have dimension "k"
print(w_train.shape,sample_train.shape)
sample_train_pca = np.transpose(w_train).dot(sample_train)
print(w_train.shape,sample_test.shape)
#sample_test_pca =  np.transpose(w_train).dot(sample_test)


# finally, build your prediction model from (sample_train_pca, label_train)
# and evaluate your model on (sample_test_pca, label_test)
# train a linear regression model using least square
# you can use libraries to implement this part (including model training and evaluating)

mod = lr()
print("-")
print(sample_train_pca.shape,label_train.shape)
print("-")
mod.fit(sample_train_pca.real, label_train)



# Plot Figure 1 based on w1 and w2
#print(w_train[0].shape)
wt1 = w_train[0].reshape(-1,1)
#wt1 = np.transpose(wt1)
wt2 = w_train[1].reshape(-1,1)
#print(wt1.shape, wt2.shape)
sample_pca_1 = mod.predict(wt1.real)  # this is a n-by-1 vector; each row is one instance and the value is its projection on w1 (1st pca feature)
sample_pca_2 = mod.predict(wt2.real) # same, but projection on w2
# now, plot data distribution based on these two features
mpl.plot(sample_pca_1,label="w1")
mpl.plot(sample_pca_2,label="w2")
mpl.title("W1 and W2")
mpl.legend()
mpl.show()
mpl.clf()



# Plot Figure 2 based on w(p-1) and wp
#sample_pca_p_1 = ...... # same, but projection on w(p-1)
#sample_pca_p = ...... # same, but projection on wp
# now plot data distribution based on these two features
