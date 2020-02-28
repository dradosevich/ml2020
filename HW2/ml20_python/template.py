# Danny Radosevich
# ML sp 20
# Work done with Mason Johnson
import numpy as np
import matplotlib.pyplot as mpl
from sklearn import linear_model
from sklearn.metrics import mean_squared_error

def alph():
    data = np.loadtxt('crimerate.csv', delimiter=',')

    alphas = [.1,.5,1,5,10,100]
    prec = [.01,.05,.1,.2,.5,.75]

    alphTrain = []
    alphTest = []
    looptest = []
    looptrain = []
    for val in alphas:
        for i in range(0,20):

            np.random.shuffle(data)
            sample = data[:,0:-1] # first p-1 columns are features
            label = data[:,-1] # last column is label

            [n,p] = sample.shape

            # start with a small training set (overfitting)
            # 0.01 - 0.5
            n_train = int(n*0.01)

            sample_train = sample[0:n_train,:]
            sample_test = sample[n_train:,:]

            label_train = label[0:n_train]
            label_test = label[n_train:]

            # first, select a model (a function with unknown parameters and learner with unknown hyper-parameters)
            # alpha can be intepreted as domain of parameters
            # 0.1 - 100
            model = linear_model.Ridge(alpha = val)

            # train the model
            model.fit(sample_train, label_train)

            # apply the model to make predictions
            label_pred_train = model.predict(sample_train)
            label_pred_test = model.predict(sample_test)


            mse_train = mean_squared_error(label_train,label_pred_train)
            mse_test = mean_squared_error(label_test,label_pred_test)

            print('\nmse_train = %f' % mse_train)
            print('mse_test  = %f' % mse_test)
            looptest.append(mse_test)
            looptrain.append(mse_train)
        alphTrain.append(np.average(looptest))
        alphTest.append(np.average(looptrain))
        looptest = []
        looptrain= []
    mpl.plot(alphTrain, label="Training")
    mpl.plot(alphTest, label="Testing")
    x = np.array([0,1,2,3,4,5])
    mpl.xticks(x, alphas)
    mpl.ylabel("Prediction Error")
    mpl.xlabel("Value of Alpha")
    mpl.title("Average of 20 tests per Alpha")
    mpl.legend()
    mpl.show()
def percent():
        data = np.loadtxt('crimerate.csv', delimiter=',')

        prec = [.01,.05,.1,.2,.5,.75]

        percTrain = []
        percTest = []
        looptest = []
        looptrain = []
        for val in prec:
            for i in range(0,20):

                np.random.shuffle(data)
                sample = data[:,0:-1] # first p-1 columns are features
                label = data[:,-1] # last column is label

                [n,p] = sample.shape

                # start with a small training set (overfitting)
                # 0.01 - 0.5
                n_train = int(n*val)

                sample_train = sample[0:n_train,:]
                sample_test = sample[n_train:,:]

                label_train = label[0:n_train]
                label_test = label[n_train:]

                # first, select a model (a function with unknown parameters and learner with unknown hyper-parameters)
                # alpha can be intepreted as domain of parameters
                # 0.1 - 100
                model = linear_model.Ridge(alpha = .1)

                # train the model
                model.fit(sample_train, label_train)

                # apply the model to make predictions
                label_pred_train = model.predict(sample_train)
                label_pred_test = model.predict(sample_test)


                mse_train = mean_squared_error(label_train,label_pred_train)
                mse_test = mean_squared_error(label_test,label_pred_test)

                print('\nmse_train = %f' % mse_train)
                print('mse_test  = %f' % mse_test)
                looptest.append(mse_test)
                looptrain.append(mse_train)
            percTrain.append(np.average(looptest))
            percTest.append(np.average(looptrain))
            looptest = []
            looptrain= []
        mpl.plot(percTrain, label="Training")
        mpl.plot(percTest, label="Testing")
        x = np.array([0,1,2,3,4,5])
        mpl.xticks(x, prec)
        mpl.ylabel("Prediction Error")
        mpl.xlabel("Value of Percentage")
        mpl.title("Average of 20 tests per Percentage")
        mpl.legend()
        mpl.show()

#alph()
percent()
