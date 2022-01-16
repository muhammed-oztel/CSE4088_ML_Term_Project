import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random


def pla(data, labels):
    w = np.zeros(31) # initialize weight vector
    itr = 0 # keep track of how many iterations until converge

    while True: # until converges
        missed = []
        n_miss = 0

        for idx, x_i in enumerate(data):
            pred = 0
            dot = np.dot(w, x_i.T)
            if dot > 0:
                pred = 1
            elif dot < 0:
                pred = -1
            else:
                pred = 0

            if pred != labels[idx]:
                n_miss += 1
                missed.append(idx)

        itr += 1
        if n_miss == 0: # if no elements missclassified pla converged
            return itr, w

        sel = np.random.choice(missed, size=1)[0] # select random point from missclassified points
        w += labels[sel]*(np.array([1, data[sel][0], data[sel][1]]))    # update weight vector  



def e_out_pla(w, target, N):    # find out of sample error for linear regression
    miss_avg_sum = 0
    for i in range(1000):
        n_miss = 0
        data = np.random.uniform(low=-1, high=1, size=(N*2)).reshape(N, 2)  # generate data

        labels = []
        labels = classify_points(data, target, labels)  # classify data

        X = np.array([(1, x[0], x[1]) for x in data]).T # restructure for bias
        y = np.array(labels).T

        pred = np.dot(w.T, X)   # predict class of points

        for i in range(N):
            if np.sign(pred[i]) != np.sign(y[i]):
                n_miss += 1
        miss_avg_sum += n_miss / N

    print(miss_avg_sum/1000)


df = pd.read_csv("data.csv")
df = df.drop(['Unnamed: 32', 'id'], axis=1)

#encoding the the target feature
df['diagnosis']= df['diagnosis'].replace('M', 1)
df['diagnosis']= df['diagnosis'].replace('B', 0)

data = df.to_numpy()

labels = data[:, 1]
data = 

sc = StandardScaler()
x_std = sc.fit_transform(x)
# X_test = sc.transform(X_test)


#splitting the dataframe and keeping 80% of the data for training and rest 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42, stratify = y)

pla_runs(x,y)