# Logistic Regression using Stochastic Gradient Descent
# Not a homework

#%%

# Logistic Regression using Stochastic Gradient Descent


import pandas as pd
from math import exp


# Need a dataset to start using functions below
# Example Dataset
dataset = pd.read_csv('/Users/kevin/Desktop/simulated_data.csv')
dataset = dataset.iloc[:,1:]
dataset = np.array(dataset)
dataset


# Make a prediction with random coefficients
def predict(row, coefficients):
	yhat = coefficients[0]
	for i in range(len(row)-1):
		yhat += coefficients[i + 1] * row[i]
	return round(1.0 / (1.0 + exp(-yhat))) # Sigmoid Function

# Randome Coefficients chosen before Gradient Descent Optimization
coef = [-0.406605464, 0.852573316, -1.104746259]

for row in dataset:
	yhat = predict(row, coef)
	print("Expected=%.3f, Predicted=%.3f [%d]" % (row[-1], yhat, round(yhat)))

####### 
# As you can see some of the expected values versus predicted values differ,
# this is why we'll now perform gradient descent, in order to optimize our "coef" values
# so that we can make better predicitions.

# Coefficient optimization for logistic regression using stochastic gradient descent
def coefficients_sgd(train, l_rate, n_epoch):
	coef = [0.0 for i in range(len(train[0]))]
	for epoch in range(n_epoch):
		sum_error = 0
		for row in train:
			yhat = predict(row, coef)
			error = row[-1] - yhat
			sum_error += error**2
			coef[0] = coef[0] + l_rate * error * yhat * (1.0 - yhat)
			for i in range(len(row)-1):
				coef[i + 1] = coef[i + 1] + l_rate * error * yhat * (1.0 - yhat) * row[i]
		print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
	return coef