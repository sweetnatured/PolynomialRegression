import pandas as pd
from sklearn.model_selection import train_test_split
from numpy.random import seed
from numpy.random import rand
import matplotlib.pyplot as plt
import random
import numpy as np

data = pd.read_csv("C:/Users/SALIH/Desktop/HomeWorks/ANNHW/cwurData.csv",skipinitialspace=True)   # Data is read
allData=data[['world_rank','quality_of_education']]

train_Data,test_Data = train_test_split(allData, test_size=0.4)



train_in=train_Data[['world_rank']]
train_out=train_Data[['quality_of_education']]
test_in=test_Data[['world_rank']]
test_out=test_Data[['quality_of_education']]


train_list_in=train_in.values.tolist()
train_list_out=train_out.values.tolist()
test_list_in=test_in.values.tolist()                #data frames converted to lists 2d
test_list_out=test_out.values.tolist()

train_data_in=np.matrix(train_list_in)
train_data_out=np.matrix(train_list_out)
test_data_in=np.matrix(test_list_in)             #Data converted to numpy matrix
test_data_out=np.matrix(test_list_out)
max=train_data_in.max()
train_data_in=train_data_in / train_data_in.max()
train_data_out = train_data_out / train_data_out.max()
test_data_in = test_data_in / max                         # DATA NORMALIZATION
test_data_out = test_data_out / max


# --------------------------DATA İS READED------------------------------------------------------------------------------

def produce_random_weights(numb):

    myList=[]

    for i in range(numb):

        my_random = random.uniform(0,1)                  # Produces Random Weights for Coefficient of Polynom
        myList.append(my_random)                         # The parameter numb indicates order of polynom

    x =np.asarray(myList)

    return myList

def calculate_forward(x,list):

    return np.polyval(x,list)                 # Creates the polynom and Calculates values

def Cost(tdo,wgh,tdi,m):

    return np.sqrt((1/2*m) * (np.sum(np.square(np.subtract(tdo,calculate_forward(wgh, tdi.tolist()))))))         #Returns Cost of model

def Gradient(tdo,wgh,tdi,m,X):

    return (1 / m) * np.sum(np.multiply(np.subtract(tdo,calculate_forward(wgh, tdi.tolist())) , X)  )               # Calculates Gradient with respect to given Weight
Cost_val= 1000000
Weights_list=[]
Costs_train=[]

for i in range(1,28):                                   # Order of model

    Weights=produce_random_weights(i)                  # Produces weights that has same number of value with order of model

    for x in range(5000):                               # Number of training

       for y in range(i):

           Weights[y] = Weights[y] -  Gradient(train_data_out,Weights,train_data_in,len(train_data_in),-(np.power(train_data_in.tolist(),len(Weights)-y))) # Weight Updating


    if(Cost(train_data_out,Weights,train_data_in,len(train_data_in))<Cost_val):    # Holds the lowest Cost

        Cost_val=Cost(train_data_out,Weights,train_data_in,len(train_data_in))
        order=i



    print("Cost of Order  ", i , "is" , Cost(train_data_out,Weights,train_data_in,len(train_data_in)))     # Prints the Cost of Model

    Costs_train.append(Cost(train_data_out,Weights,train_data_in,len(train_data_in)))
    Weights_list.append(Weights)    # Adds the updated weights to list in order to use at Testing


print("Minimum Cost of Polynoms is ", order, " Order ", Cost_val)    # Prints the order of minimum cost

print("-----------------------Training Completed-------------------------------")

Cost_test=[]
for i in Weights_list:

    print("Test Error of ",len(i) ," degree is", Cost(test_data_out,i,test_data_in,len(test_data_out)))         # Tests the data with respect to given weights from training
    Cost_test.append(Cost(test_data_out,i,test_data_in,len(test_data_out)))

plt.plot(range(1,28),Costs_train)
plt.plot(range(1,28),Cost_test)
plt.show()

plt.plot(range(1,28),Costs_train)
plt.show()
plt.plot(range(1,28),Cost_test)
plt.show()

