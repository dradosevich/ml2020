
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsRegressor as KNR
import matplotlib.pyplot as plt

# this is the COMPAS data set (recidivism prediction)
data = pd.read_csv("compas.csv")

# drop duplicate rows
data.drop_duplicates(keep = False, inplace = True)
data = np.asarray(data)

sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

# let us use 75% for training
n_train = int(n*0.75)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

trainN, trainP = sample_train.shape
testN, testP = sample_test.shape

label_train = label[0:n_train]
label_test = label[n_train:]

# now, build your kNN classifier (or, kkNN classifier)...
# basically, you may want to first compute a n-by-m distance matrix,
# where n is the number of training instances and m is number of testing instances
"""
print(type(trainN),type(testN))
knn =np.zeros((trainN,testN),dtype=np.int)
#......
i =0;
dicts=[]
for a in sample_test:
    j = 0
    dic = {}
    for b in sample_train:
        temp = np.linalg.norm(a-b)
        knn[j,i]= temp
        if temp not in dic:
            dic[temp] = j
        j+=1
    dicts.append(dic)
    i+=1
#......

#......

# now get your testing error, store it for different K, and prepare to plot a figure
terr = []
for k in ks:
    for dict in dict
"""
terr = []
ks = [1,3,5,7,9]
for k in ks:
    knn = KNR(n_neighbors = k)
    knn.fit(sample_train,label_train)
    pred = knn.predict(sample_test)
    mse = (((pred - label_test) ** 2).sum()) / len(pred)
    terr.append(mse)

plt.plot(terr)
plt.title("KNN Testing error")
plt.legend()
plt.xticks([])
plt.show()

# ......

# ......
