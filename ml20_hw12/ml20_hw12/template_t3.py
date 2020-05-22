
import pandas as pd 

# this is the COMPAS data set (recidivism prediction)
data = pd.read_csv("compas.csv") 

# drop duplicate rows
data.drop_duplicates(keep = False, inplace = True) 

sample = data[:,0:-1] 
label = data[:,-1] 
[n,p] = sample.shape

# choose your own percentage of training data 
n_train = int(n*_____)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

# now, implement your Laplacian mechanism
# basically, you first perturb the probability of your LR model 
# and then threshold it to make classification 

#......

#......

#......

# now plot the curve of testing error versus different epsilson 

# ......

# ......

