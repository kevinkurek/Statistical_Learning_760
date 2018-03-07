# HW5 Statistical Learning 760

# Perceptron Learning Algorithm & Gradient Descent

# Told to create a positive and negative class artificially which is
# linearly separable, then write the perceptron learning algorithm that
# cuts the two classes


#%%
import random
import numpy as np
import matplotlib.pyplot as plt

number_of_observations = 100
x1 = np.random.multivariate_normal([0,0], [[1,.75],[.75,1]], number_of_observations)
x2 = np.random.multivariate_normal([2,5], [[1,.75],[.75,1]], number_of_observations)

#%%

# Simulated Input vectors pushed together
simulated_data = np.vstack((x1,x2)).astype(np.float32)

# Simulated Target vector made of 0's and 1's
simulated_labels = np.hstack((np.zeros(number_of_observations),(np.ones(number_of_observations))))

# Plotting simulated data for visual with graph
plt.figure(figsize=(12,8))
plt.title('Simulated Data for Perceptron Learning Algorithm')
plt.scatter(simulated_data[:,0],simulated_data[:,1],c=simulated_labels, alpha=.8)
plt.show()

# Turning Simulated data into Dataframes
simulated_data_df = pd.DataFrame(simulated_data)
simulated_labels_df = pd.DataFrame(simulated_labels).astype(int)

# Concatenating input and predictor columns together in single dataframe for visual ease
all_simulated_data_df = pd.concat([simulated_data_df,simulated_labels_df],axis=1)
all_simulated_data_df.columns = ['x1','x2','predictor']
all_simulated_data_df.head()
all_simulated_data_df.to_csv("/Users/kevin/Desktop/simulated_data.csv")
###############################################################

#%%
all_simulated_data_array = np.array(all_simulated_data_df)
# all_simulated_data_array[2].astype(int)
print(all_simulated_data_df.iloc[1,2].astype(int)) # shape for row is 3x1 or (3,) array

#%%
# Initialization of weights to zeros
weights = np.zeros((3,1))
print(weights.shape)

#%%
# Array dot product
#print(np.dot(weights.T,all_simulated_data_df[0]))


######## Need dot product coming from first two columns in df
#%%
# weights = np.random.random((3,1))
# #weights = np.random.rand(1,3)
# print(weights)


#%%
######## Testing for function below
# weights = [0.1,0.1,0.1]

# all_simulated_data_array = np.array(all_simulated_data_df)

# for row in all_simulated_data_array:
#   #print(row)
#   if (row[-1] == 1) & (np.dot(weights,row) >= 0):
#     print(row)
#     #print(np.dot(weights,row))

# # for index,row in all_simulated_data_array.iterrows():
# #   print(row)
#   # if x[-1] == 1:
#   #    if np.dot(weights.T,x) >= 0:




#%%

# Initialize weight vector to any number
weights = [0.1,0.1]

def Perceptron_Learning_Algorithm(input_array,weights):
  t = 0
  for row in input_array:
    if (row[-1]==1) & (np.dot(weights,row[:-1]) >= 0):
      continue
    elif (row[-1]==1) & (np.dot(weights,row[:-1]) <= 0):
      weights = weights + row[:-1]
      t = t + 1
    elif (row[-1]==0) & (np.dot(weights,row[:-1]) < 0):
      continue
    elif (row[-1]==0) & (np.dot(weights,row[:-1]) >= 0):
      weights = weights - row[:-1]
      t = t + 1
  return weights, t
weights,t = Perceptron_Learning_Algorithm(all_simulated_data_array,weights)
# print(weights,t)


#####################################################
# Perceptron Gradient Descent
#####################################################

#%%
# Data
all_simulated_data_array = np.array(all_simulated_data_df)
dataset = all_simulated_data_array
# print(dataset)


# Start Gradient Descent & Algorithm
from random import randrange
import numpy as np
import pandas as pd


def predict(row, weights):
	activation = weights[0]
	for i in range(len(row)-1):
		activation += weights[i + 1] * row[i]
	return 1.0 if activation >= 0.0 else 0.0

#weights = [-0.1, 0.20653640140000007, -0.23418117710000003]
##### Testing predict function
# for row in dataset:
# 	prediction = predict(row, weights)
# 	print("Expected=%d, Predicted=%d" % (row[-1], prediction))

# Estimate Perceptron weights using stochastic gradient descent
def train_weights(train, l_rate, n_epoch):
	weights = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0.0
		for row in train:
			prediction = predict(row, weights)
			error = row[-1] - prediction
			sum_error += error**2
			weights[0] = weights[0] + l_rate * error
			for i in range(len(row)-1):
				weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
		#print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return weights

##### Testing train_weights function
weights = train_weights(dataset,0.1,5)
print('X1 weight = %.3f, X2 weight = %.3f' % (weights[1],weights[2]))

# Generate linear separator based on weights
x = np.linspace(-3,5,100)
#print(x)

# Generate linear separator based on weights
plt.figure(figsize=(12,8))
plt.title('Simulated Data for Perceptron Learning Algorithm')
#plt.scatter(weights[1],weights[2])
plt.plot(weights[1]*x,weights[2]*x,label='Weight Vector Span')
#plt.scatter(-1,(-weights[1]/weights[2]),c='r')
plt.plot(x,(-4.95*weights[1]/weights[2])*x+.89,label='Linearly Separator')
plt.scatter(simulated_data[:,0],simulated_data[:,1],c=simulated_labels, alpha=.8)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()