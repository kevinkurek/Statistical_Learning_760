# HW4 Part 2 Statistical Learning 760

# Logisitic Regression for Digits

#%%
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import operator

df_0 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.0.txt',header=None)
df_0['Digit'] = 0
df_3 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.3.txt',header=None)
df_3['Digit'] = 3
df_2 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.2.txt',header=None)
df_2['Digit'] = 2
df_7 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.7.txt',header=None)
df_7['Digit'] = 7
df_1 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.1.txt',header=None)
df_1['Digit'] = 1
df_9 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.9.txt',header=None)
df_9['Digit'] = 9

#%%
################
# Alternative way to load from txt file into len(rows) x len(features) array
################
def _load_training_data(file_name):
    features = []

    with open(file_name, 'r') as file:
        for row in file:
            data_string = row.strip().split(',')
            data = []
            for i in range(len(data_string)):
                data.append(float(data_string[i]))

            features.append(data)

    return np.array(features)

training_data_0 = _load_training_data("/Users/kevin/Desktop/Stats 760/Homework1/train.0.txt")
training_data_3 = _load_training_data("/Users/kevin/Desktop/Stats 760/Homework1/train.3.txt")
print(training_data_0.shape) # 1194 x 256 array
print(training_data_0)
print(training_data_3) # n x 256 array



#%%
class Logistic_Regression_algo:

  def __init__(self,df_list):
    self.df_list = df_list

  # Concatentate digit dataframes of interest together
  def concat_dfs(self):
    self.df = pd.concat(self.df_list)
    #print(len(self.df))
    return self.df

  # Subsetting to input vector X, and output value y.
  def features(self):
    self.X = self.df.iloc[:,:-1]
    self.y = self.df.iloc[:,-1]
    self.X = np.array(self.X) # Converting to array
    self.y = np.array(self.y) # Converting to array
    return self.X,self.y

  def logistic_model(self):

    # Need to code the linear model to have the regression parameter
    # separate the two classes of digits
    cv_score = []
    self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=.3, random_state=7,stratify=self.y)
    self.model = LogisticRegression()
    scores = cross_val_score(self.model,self.X_train,self.y_train,cv=10,scoring='neg_mean_squared_error') # 10-fold Cross Validation, but scoring is mean_squared_error for logistic regression metric
    cv_score.append(scores.mean())
    return 'It worked'

  def prediction(self):
    self.model = LogisticRegression()
    self.model.fit(self.X_train,self.y_train)
    self.y_pred = self.model.predict(self.X_test)
    print(confusion_matrix(self.y_test, self.y_pred.round())) # Had to round self.y_pred in order to round to nearest number it was predicting
    return classification_report(self.y_test,self.y_pred.round())

first = Logistic_Regression_algo([df_0,df_3])
first.concat_dfs()
first.features()
print(first.logistic_model())
print(first.prediction())

second = Logistic_Regression_algo([df_2,df_7])
second.concat_dfs()
second.features()
print(second.logistic_model())
print(second.prediction())

third = Logistic_Regression_algo([df_1,df_9])
third.concat_dfs()
third.features()
print(third.logistic_model())
print(third.prediction())




#%%

################################
# Example graph using simulated data from multivariable normal distribution
################################
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(12)
num_observations = 5000

x1 = np.random.multivariate_normal([0, 0], [[1, .75],[.75, 1]], num_observations)
x2 = np.random.multivariate_normal([1, 4], [[1, .75],[.75, 1]], num_observations)

simulated_separableish_features = np.vstack((x1, x2)).astype(np.float32)
simulated_labels = np.hstack((np.zeros(num_observations), np.ones(num_observations)))
print(type(simulated_separableish_features))
print(simulated_separableish_features)
print(type(simulated_labels))
print(simulated_labels)

plt.figure(figsize=(12,8))
plt.scatter(simulated_separableish_features[:, 0], simulated_separableish_features[:, 1],
            c = simulated_labels, alpha = .4)


#%%
################################
# Example Logistic Regression using simulated data AND digits data
# from above which returns coefficient as well as intercepts 
################################

#%%

class clean:
  """ Input: List of df's from digits data 
      Output: Returns X input features and y target variable"""
  def __init__(self,df_list):
    self.df_list = df_list

  # Concatentate digit dataframes of interest together
  def concat_dfs(self):
    self.df = pd.concat(self.df_list)
    #print(len(self.df))
    return self.df

  # Subsetting to input vector X, and output value y.
  def features(self):
    self.X = self.df.iloc[:,:-1]
    self.y = self.df.iloc[:,-1]
    self.X = np.array(self.X) # Converting to array
    self.y = np.array(self.y) # Converting to array
    return self.X,self.y
first = clean([df_0,df_1])
first.concat_dfs()
X, y = first.features()

# Defining a sigmoid function
def sigmoid(scores):
  return 1 / (1+np.exp(-scores))

# Defining log likelihood function
def log_likelihood(features,target,weights):
  scores = np.dot(features,weights)
  log_like = sum(target*scores - np.log(1+np.exp(scores)))
  return log_like

# Defining Logistic_Regression with weights
def logistic_regression(features, target, num_steps, learning_rate, add_intercept = False):
    """ Input: feature array (input variables), target variable, number of steps to take,
                learning rate that you define, intercept or no intercept (default is false)
        Output: Prints periodic log_likelihood calculated from function above
                Returns weights of coefficients"""
    if add_intercept:
        intercept = np.ones((features.shape[0], 1))
        features = np.hstack((intercept, features))
        
    weights = np.zeros(features.shape[1])
    
    for step in range(num_steps):
        scores = np.dot(features, weights)
        predictions = sigmoid(scores)

        # Update weights with log likelihood gradient
        output_error_signal = target - predictions
        
        gradient = np.dot(features.T, output_error_signal)
        weights += learning_rate * gradient

        # Print log-likelihood every so often
        if step % 10000 == 0:
            print(log_likelihood(features, target, weights))
        
    return weights

# Simulated logistic regression on 2 input variables
weights = logistic_regression(simulated_separableish_features, simulated_labels,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
print(len(weights)) # 3 weights because of intercept

# Digits logistic regression on 256 input variables
weights_1 = logistic_regression(X, y,
                     num_steps = 50000, learning_rate = 5e-5, add_intercept=True)
print(len(weights_1)) # 257 weights because of intercept

#%%
from sklearn.linear_model import LogisticRegression

def sklearn_logistic_regression(features,target):
  """Input: features matrix/array, target vector/array
      Output: Returns intercept and coefficients 
      
      SHOULD BE ALMOST EXACT SAME COEFFICIENTS AS ABOVE BUT SLIGHTLY"""

  clf = LogisticRegression(fit_intercept=True,C=1e15)
  clf.fit(features,target)
  return clf.intercept_, clf.coef_
print(sklearn_logistic_regression(simulated_separableish_features,simulated_labels))
print(sklearn_logistic_regression(X,y))