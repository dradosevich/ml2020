#Danny Radosevich
# you cannot use PCA library
#STEPS FROM LECTURE
#Estimate SIGMA_x
#SIG_x = (1/n)(SIG1-n)(Xi-mew)(xi-mew)^T where mew = (1/n) SIG(1-n)x_i
#Apply package to get eigenvectors w
#Also get assosciated values
#eigvals, eigvecs = np.linalg.eig(SIG)

import numpy as np
import matplotlib.pyplot as mpl

#
data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape



# let's learn PCA vectors from the entire sample set (i.e., "sample")
# note that label is not used -- therefore PCA is an "unsuperivsed learning" technique
# cannot use PCA libraries; can use libraries to find eigenvectors/eigenvalues of a matrix
def getMew():
    n = 0
    return (np.mean(sample)
def getSigma():
    #stuff und things
    mew = getMew()

    xmew = sample-mew
    trans = np.transpose(xmew)
    sig = xmew.dot(trans)
    sig = np.asarray(sig)
    return (np.mean(sig))



# you will find p number of PCA projection vectors w1, w2, ..., wp
# we stypically store them in a p-by-k matrix "w"; each column being one vector
# typically, vectors are sorted in a way that, 1st column is the optimal, 2nd column is the 2nd optimal, etc
# tip: many libraries will automatically sort eigenvectors based on their eigenvalues
#w = .....

#print(getMew().shape)
#print(sample.shape)

sig = getSigma() #start off by getting our sigma
eigvals, eigvecs = np.linalg.eig(sig) #eigvecs is the w vector
print(eigvals.shape,eigvecs.shape)
print(np.argmax(eigvals))
#now to utilize w_1 and w_2
# Plot Figure 1 based on w1 and w2
sample_pca_1 = eigvecs[0] # this is a n-by-1 vector; each row is one instance and the value is its projection on w1 (1st pca feature)
sample_pca_2 = eigvecs[1] # same, but projection on w2
print(sample_pca_1.shape,sample_pca_2.shape)
mpl.plot(sample_pca_1,label="w1")
mpl.plot(sample_pca_2,label="w2")
mpl.title("W1 and W2")
mpl.legend()
mpl.show()
mpl.clf()

# Plot Figure 2 based on w(p-1) and wp
sample_pca_p_1 = eigvecs[len(eigvecs)-2] # same, but projection on w(p-1)
sample_pca_p = eigvecs[len(eigvecs)-1] # same, but projection on wp
# now plot data distribution based on these two features
mpl.plot(sample_pca_p_1,label="w1")
mpl.plot(sample_pca_p,label="w2")
mpl.title("W1 and W2")
mpl.legend()
mpl.show()
