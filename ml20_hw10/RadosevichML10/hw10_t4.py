
# Danny Radosevich
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture as GMM
from sklearn.decomposition import PCA
#
data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape


# apply GMM to cluster "sample" (same K as in task [1])
gmm = GMM(n_components = 3).fit(sample)
cluster = gmm.predict(sample)
# project "sample" into a 2D feature space obtained by PCA
# if you are confident about your own implementation in hw9, use it
# otherwise, you can use PCA library
pca = PCA(n_components=2)
test=pca.fit_transform(sample)
plt.scatter(test[:,0],test[:,1],c=cluster)
plt.show()
# color your instances based on your clustering results
