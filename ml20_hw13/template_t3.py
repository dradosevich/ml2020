
import pandas as pd 

# this is the COMPAS data set (recidivism prediction)
data = pd.read_csv("compas.csv") 

# drop duplicate rows
data.drop_duplicates(keep = False, inplace = True) 

sample = data[:,0:-1] 
label = data[:,-1] 
[n,p] = sample.shape

# choose 50% for training 
n_train = int(n*0.5)

sample_train = sample[0:n_train,:]
sample_test = sample[n_train:,:]

label_train = label[0:n_train]
label_test = label[n_train:]

# now, implement bagging of logistic regression

# 1. choose a regularization coefficient
# ......

# 2. choose a size of bootstrap sample 
# ......

# 3. choose the number of base models 
# ......

# 4. implement bagging
# ......
# ......
# ......

# 5. evaluate testing error
# ......

# 6. plot error curve versus different number of base models 
# ......