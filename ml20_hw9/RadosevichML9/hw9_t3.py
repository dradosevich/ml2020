
# you cannot use PCA library
import numpy as np
import matplotlib.pyplot as mpl
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
    return (np.mean(np.transpose(sample_train),axis=1))

def getSigma():
    #stuff und things
    mew = getMew()

    print("getSig\n",sample_train.shape, mew.shape)
    xmew = sample_train-mew
    print(xmew.shape)
    trans = np.transpose(xmew)
    sig = trans.dot(xmew)
    print(sig.shape)
    #sig = np.asarray(sig)
    return (sig.dot(1/sig.shape[0]))


# store your k projection vectors in a matrix "w_train" (similar to w in [2])
# note: here you need to vary k to get different prediction mse

sig = getSigma()
print(sig.shape)
evals, emat = np.linalg.eig(sig)
print("------------------------")
print(evals.shape,emat.shape)
print("------------------------")
#w_train = np.transpose(sig).dot(sample_train)
w_train = emat
print("xtild\n",w_train.shape,sample_train.shape)
xtild = np.dot(w_train,np.transpose(sample_train))
print(xtild.shape)



# next, project both training sample and testing sample onto w_train
# both training and testing instances will now have dimension "k"
print(w_train.shape,sample_train.shape)
#sample_train_pca = np.transpose(w_train).dot(sample_train)
sample_train_pca = np.dot(sample_train,w_train)
print("-------",w_train.shape,sample_test.shape)
#sample_test_pca =  np.transpose(w_train).dot(sample_test)
sample_test_pca = np.dot(sample_test,w_train)

# finally, build your prediction model from (sample_train_pca, label_train)
# and evaluate your model on (sample_test_pca, label_test)
# train a linear regression model using least square
# you can use libraries to implement this part (including model training and evaluating)
testk=w_train;
test=[]
for i in range(99):
    mod = lr()
    print("-")
    print(sample_train_pca.shape,label_train.shape)
    print("-")
    mod.fit(np.dot(sample_train, testk), label_train)

    # Plot Figure 1 based on w1 and w2
    #print(w_train[0].shape)
    #wt1 = w_train[0]#.reshape(-1,1)
    #wt1 = np.transpose(wt1)
    #wt2 = w_train[1]#.reshape(-1,1)
    #print(wt1.shape, wt2.shape)
    sample_pca_1 = mod.predict(np.dot(sample_test,testk))  # this is a n-by-1 vector; each row is one instance and the value is its projection on w1 (1st pca feature)
    #sample_pca_2 = mod.predict(np.dot(sample_test,wt2)) # same, but projection on w2
    error = np.square(sample_pca_1-label_test).mean()
    test.append(error)
    testk=testk[:,0:-1]

# now, plot data distribution based on these two features
test.reverse()
mpl.plot(test,label="MSE Error")
#mpl.plot(sample_pca_2,label="w2")
#mpl.title("W1 and W2")
mpl.ylabel("MSE")
mpl.xlabel("Number of instances used")
mpl.legend()
mpl.show()
mpl.clf()



# Plot Figure 2 based on w(p-1) and wp
#sample_pca_p_1 = ...... # same, but projection on w(p-1)
#sample_pca_p = ...... # same, but projection on wp
# now plot data distribution based on these two features
