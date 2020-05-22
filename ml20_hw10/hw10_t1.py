#Danny Radosevich

import numpy as np
import random
import math
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

#
data = np.genfromtxt('crimerate.csv', delimiter=',')
sample = data[:,0:-1]
label = data[:,-1]
[n,p] = sample.shape

#TEST K--------------------------------
testk = 9
sevenksone=[1,3,5,7,9,11,13]

#TEST K--------------------------------

#k averages --------------------------------
kaverage = []
#k averages --------------------------------

#j averages --------------------------------
jone = []
jtwo = []
#j averages --------------------------------
# implement K-means from scratch. you can use library to measure L2 distance.
# apply K-means on "sample"

def clusterPoints(mew):
    mews = {}
    #print(type(mews))

    for x in sample:
        #testmews = []
        bestmew = (math.inf, math.inf)
        i = 0
        key = 0
        for k in range(len(mew)):
            #print(type(m),"*")
            temp = np.linalg.norm(x-mew[k],ord=2)
            #testmews.append(temp)
            if temp < bestmew[1]:
                bestmew = (i,temp)
                key = k
            i+=1
        #key = min(enumerate(testmews))[0]
        #key = bestmew[0]
        #print(type(mew),len(mew))
        #print(key)
        if key in mews: #check if we already have the best mew key
            #print(type(mews[bestmew]))
            mews[key].append(x) #append the new x to mew cluster
        else:
            mews[key] = [x] #add the best mew key and its starter x
        i+=1

    #print("++\nCluster Points",len(mew),len(mews),"\n++")
    return mews

def getMew(mew, clusters):
    #print("*",len(clusters))
    mewtwo = []
    #print(type(clusters))
    keys = sorted(clusters.keys())
    #print(type(keys))
    for key in keys:
        mewtwo.append(np.mean(clusters[key],axis = 0))
    return mewtwo

def converged(mew, mewtwo):
    #print("------")
    #check if mismatched lengths
    if(len(mew)!=len(mewtwo)):
        #print(len(mew),len(mewtwo))
        return False
    for i in range(len(mew)):
        a = mew[i]
        b = mewtwo[i]
        #print(type(a))
        #print(type(b))
        #print(a)
        #print(b)
        if not(a == b).all():
            return False
    #check to see if elems match, has converged
    #for i in range(len(mew)):
    #default return statement
    return True

def startMew(k):
    mew = {}
    toreturn = []
    for x in sample:
        group = random.randint(1,k)
        if group in  mew:
            mew[group].append(x)
        else:
            mew[group] = [x]
    for k in range(1,k+1):
        #print(np.mean(mew[k],axis=0))
        toreturn.append(np.mean(mew[k],axis = 0))
    return toreturn

def kmeans(k):
    #find k clusters based on the supplied
    mew = startMew(k) #start with random mew
    #mew = random.sample(list(sample),k)
    #print(len(mew))
    mewtwo = startMew(k) #start with random mew
    #mewtwo = random.sample(list(sample),k)
    clusters = []
    i = 0
    line = ""
    while not converged(mew, mewtwo): #see if need to continue
        #print("kmeans while",len(mewtwo))
        print(i)
        i+=1
        #print("test")
        mew = mewtwo #transition mews
        clusters = clusterPoints(mewtwo)
        #print(clusters)
        mewtwo = getMew(mewtwo,clusters) #get the new mew
        #print(mewtwo[0:1][0:1])
        #print("----------------------------------------------------------------")
    return(mewtwo, clusters)

def plot():
    mewtwo,clusters = kmeans(testk)
    #print(len(mewtwo))
    #print(len(clusters))
    # project "sample" into a 2D feature space obtained by PCA
    # if you are confident about your own implementation in hw9, use it
    # otherwise, you can use PCA library
    #print(type(mewtwo))
    pca = PCA(n_components=2)
    mewtwo = np.asarray(mewtwo)
    mewtwo=pca.fit_transform(mewtwo)
    plt.scatter(mewtwo[:,0],mewtwo[:,1])
    #plt.show()

    clustersizes = []
    for clust in range(len(clusters)):
        clustersizes.append(len(clusters[clust]))
    #print(clustersizes)


    projected = clusters[0]
    for i in range(1,len(mewtwo)):
        projected = np.concatenate((projected,clusters[i]))


    projected = pca.fit_transform(projected)

    i=0
    for clust in clustersizes:
        clust = clust+i
        plt.scatter(projected[i:clust,0],projected[i:clust,1])
        i=clust
    #plt.scatter(projected[:,0],projected[:,1])
    #for i in range(0,testk):
        #projected = pca.fit_transform(clusters[i])
        #plt.scatter(projected[:,0],projected[:,1])
    plt.show()
def getJ(x, mew):
    #get the j, to be alters
    return np.linalg.norm(x-mew,ord=2)
def average():
    for k in sevenksone:
        mewtwo,clusters = kmeans(k)
        for i in range(len(clusters)):
            j = getJ(mewtwo[i],clusters[i])
            print(type(j))
            jone.append(j*(1/len(mewtwo[i])))
            jtwo.append(j)
    plt.plot(jone,label="j1")
    plt.title("J1")
    plt.legend()
    plt.xticks([])
    plt.show()
    plt.clf()
    plt.xticks([])
    plt.plot(jtwo,label="J2")
    plt.title("J2")
    plt.legend()
    plt.show()
    plt.clf()
# color your instances based on your clustering results
#plot()
average()
