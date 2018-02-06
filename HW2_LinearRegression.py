# HW2 Statistical Learning 760

# Linear Regression Classifier for digits
# and
# Prostate cancer k-subset optimization for linear Regression

#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
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


class linear_regression_algo():

  def __init__(self,df_list):
    self.df_list = df_list

  # Concatentate digit 0 and digit 3 dataframes together
  def concat_dfs(self):
    self.df = pd.concat(self.df_list)
    #print(len(self.df))
    return self.df

  # Subsetting to input vector X, and output value y.
  def features(self):
    self.X = self.df.iloc[:,:-1]
    self.y = self.df.iloc[:,-1]
    return self.X,self.y

  def linear_model(self):

    # Need to code the linear model to have the regression parameter
    # separate the two classes of digits
    cv_score = []
    self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=.3, random_state=7,stratify=self.y)
    self.regression = linear_model.LinearRegression()
    scores = cross_val_score(self.regression,self.X_train,self.y_train,cv=10,scoring='neg_mean_squared_error') # 10-fold Cross Validation, but scoring is mean_squared_error for linear regression metric
    cv_score.append(scores.mean())
    return 'It worked'

  def prediction(self):
    self.regression = linear_model.LinearRegression()
    self.regression.fit(self.X_train,self.y_train)
    self.y_pred = self.regression.predict(self.X_test)
    print(confusion_matrix(self.y_test, self.y_pred.round())) # Had to round self.y_pred in order to round to nearest number it was predicting
    return classification_report(self.y_test,self.y_pred.round())

first = linear_regression_algo([df_0,df_3])
first.concat_dfs()
first.features()
print(first.linear_model())
print(first.prediction())

second = linear_regression_algo([df_2,df_7])
second.concat_dfs()
second.features()
print(second.linear_model())
print(second.prediction())

third = linear_regression_algo([df_1,df_9])
third.concat_dfs()
third.features()
print(third.linear_model())
print(third.prediction())








# Prostate cancer k-subset optimization for linear Regression

#%%
import pandas as pd
from sklearn import linear_model

df = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework2/prostate.txt',sep='\t') #tab delimited
# print(df.head())
df = df.drop(df.columns[0],axis=1) # First column was a duplicate index
df.head()

# Splitting into same training and test set used in the data
training_df = df[df['train']=='T']
test_df = df[df['train']=='F']
training_df = training_df.drop('train', axis=1)
test_df = test_df.drop('train', axis=1)
assert len(df)==(len(training_df)+len(test_df)), 'Lengths dont match original data' # Didn't show, so they match

#%%
# Split training_df into X and y components
X_train = training_df.iloc[:,:-1]
y_train = training_df.iloc[:,-1]
print(X_train)
print(y_train)


# STOPPED HERE
# Design Single Linear Regression Algorithm
# Design a loop through all the combinations of possible linear regeressions

#%%
regression = linear_model.LinearRegression()
regression.fit(training_df,test_df)