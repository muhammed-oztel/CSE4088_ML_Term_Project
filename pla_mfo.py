import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import random

def target_function(): 
    # generate random 4 points and reshaping it to work easily
    target = np.random.uniform(-1,1, size=4).reshape(2,2)
    return target



def find_missclassified(weight,inserted, sign_list):
    # declareing an empty list to add missclassified points
    missclassified = []
    # checking every point in dataset whether it is correctly classified or not
    for i in range(len(inserted)):
        # matrice multiplication with weight and point to find where the point stands
        product = np.matmul(weight, inserted[i]) 
        g_sign = 0 
        if product<0:g_sign = -1
        elif product>0:g_sign= 1
        # checking whether the point is correctly classified or not
        # if not add to miss list
        if sign_list[i] !=g_sign:
            missclassified.append(i)
    return missclassified

def perceptron_la(weight, inserted, sign_list):
    counter = 1
    while True:
        # Find all missclassified points
        missclassified = find_missclassified(weight,inserted, sign_list)
        # if there is no missclassified point means its fine, break the loop
        if missclassified == []:
            break
        # take random point from missclassifieds
        rand_choice = random.choice(inserted[missclassified])
        # y_sign = find_boundary(rand_choice[1],rand_choice[2],line) 
        weight += sign_list * rand_choice # update the weight vector based on PLA formula
        counter +=1 # increment the counter
    return counter


def run_with_iter(dataset, target, num_of_iters):
    list_of_counters = [] 

    for i in range(num_of_iters):

        # dataset = np.random.uniform(-1,1, size=num_of_data*2).reshape(num_of_data,2)
        inserted = np.insert(dataset,0,1,axis=1) 
        weight = np.zeros(len(dataset[0])) 
        count= perceptron_la(weight,inserted, target) 
        list_of_counters.append(count) 
    return list_of_counters, weight
    
def find_error( num_of_iters, num_of_fresh, weight, line ):
    fresh_probs = [] # list of probability of newly created point
    # loop for generating N times
    for i in range(num_of_iters):
        # generate new points to check with target function and weight of last run
        fresh_points = np.insert(np.random.uniform(-1,1, size=num_of_fresh*2).reshape(num_of_fresh,2),0,1,axis=1)
        signs_of_fresh = [] # list of new point signs
        for i in range(len(fresh_points)): # iterating over every point to find sign and add to list
            signs_of_fresh.append(find_boundary(fresh_points[i][1],fresh_points[i][2],line))
        predict = np.sign(np.matmul(weight, fresh_points.T)) # checking if the prediction is correct with matrice multiplication

        miss_count = 0
        for i in range(len(predict)): # counting how many miss predictions occured
            if int(predict[i]) != int(signs_of_fresh[i]):miss_count+=1
        fresh_probs.append(miss_count / len(fresh_points)) # ratio of miss over size of dataset
    return np.average(fresh_probs) # average of final predictions


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


#splitting the dataframe and keeping 80% of the data for training and rest 20% for testing
# X_train, X_test, y_train, y_test = train_test_split(x_std, y, test_size=0.2, random_state=42, stratify = y)

pla_runs(x,y)