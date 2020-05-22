
import pandas as pd 

# this is the crime rate data set(predict continuous crime rate)
data = pd.read_csv("crimerate.csv") 
sample = data[:,0:-1] 
label = data[:,-1] 
[n,p] = sample.shape

# we will divide data into three sets: two training (75%) and one testing (25%)  
n_train = int(n*0.75)
# Set 1 is the initial training set (which we use to initially train beta by standard LS)
# here I choose 10 instances, but feel free to play with it to get smoother and convergent error curve 
n_train_initial = 10 
# Set 2 is the sequential training set (which we use to update beta iteratively, each time using one instance)
# it contrains the remaining training instances 
n_train_seq = n_train - n_train_initial 
# Set 3 is the testing set (which we keep evaluating beta on) 
n_test = n - n_train

# construct the actual sets 
sample_train_initial = sample[0:n_train_initial,:]
sample_train_seq = sample[n_train_initial:n_train,:]
sample_test = sample[n_train:,:]

label_train_initial = label[0:n_train_initial]
label_train_seq = label[n_train_initial:n_train]
label_test = label[n_train:]

# now, implement stochastic gradient descent to optimize beta

# Step 1. Initially train beta using "sample_train_initial" by least square 
# you can implement this yourself or use library 
# ......

# Step 2. Loop. Every time uses an instance from "sample_train_seq" to update beta 
# after every update, evaluate the new beta on "sample_test" to get a testing error 
# ......

# Step 3. Plot testing error curve versus nubmer of updates 
# ......
