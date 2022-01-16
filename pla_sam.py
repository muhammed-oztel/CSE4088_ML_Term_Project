import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def points_classifer(d, random_line_points):
    my_list = []
    for i in d:
        x = i[0]
        y = i[1]
        ax , ay = random_line_points[0][0], random_line_points[0][1]
        bx , by = random_line_points[1][0], random_line_points[1][1]
        ret = ((bx - ax) * (y - ay) - (by - ay) * (x - ax))
        if ret > 0: 
            my_list.append(1)
        elif ret <= 0: 
            my_list.append(-1)
    return my_list

def pla(d, y):
    weight = np.zeros(len(d[0]))
    counter = 0
    while True:
        misclassifed_index = []
        for i, xi in enumerate(d):
            # input = np.array([1, xi[0] , xi[1]])
            # input = np.insert(xi,0,1)
            dot = np.dot(weight, xi.T) 
            if dot <= 0:
                val = -1
            else:
                val = 1
            
            if val != y[i]:
                misclassifed_index.append(i)
        print("misclassified points:", len(misclassifed_index))
        counter +=1
        if len(misclassifed_index) < 10:
            return counter, weight
            # return number of iternations 
        rand_ind = np.random.choice(misclassifed_index, 1)[0]
        # print(rand_ind)
        # np.insert(d[rand_ind],0,1)
        weight += y[rand_ind] * d[rand_ind]
        # weight += y[rand_ind] * np.array([1, d[rand_ind][0],d[rand_ind][1]])

def pla_runs(x,y):
    N = 100

    # testing_set = np.random.uniform(-1, 1, 2*N).reshape(N,2)

    # training_set = np.random.uniform(-1, 1, 2*N).reshape(N,2)
    
    # target_function = np.random.uniform(-1, 1, 4).reshape(2,2)
    # Y = points_classifer(training_set, target_function)

    res, w = pla(x, y)
    
    print(res)
    print(w)
    



df = pd.read_csv("data.csv")
df = df.drop(['Unnamed: 32', 'id'], axis=1)
# df= df.iloc[1:, :]

#encoding the the target feature
df['diagnosis']= df['diagnosis'].replace('M', 1)
df['diagnosis']= df['diagnosis'].replace('B', 0)
# y = df.iloc[1:,:1]
data = df.values

x = data[:,1:]
y = data[:,:1]

sc = StandardScaler()
x_std = sc.fit_transform(x)
# X_test = sc.transform(X_test)

# print(x)
#splitting the dataframe and keeping 80% of the data for training and rest 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42, stratify = y)
res, w = pla(x_std, y)
    
print(res)
print(w)
# pla_runs(x,y)