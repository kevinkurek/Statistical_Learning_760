# HW3 Statistical Learning 760

# Principal Component Analysis fed into a
# Gaussian Process Classifier for digits

#%%
import numpy as np
import pandas as pd
from sklearn import linear_model
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

class data_clean():

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

  # Dimension Reduction using PCA for feature selection/retention
  def Principal_Components():
  # Want to create a principal component analysis and 
  # take the "n" most relevant features

  # Defining Train & Splitting of data on Principal Components
  def train_test(self):

    self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=.3, random_state=7,stratify=self.y)
    return self.X_train, self.X_test, self.y_train, self.y_test

clean = data_clean([df_0,df_3,df_2,df_7,df_1,df_9])
clean.concat_dfs()
X, y = clean.features()
X_train, X_test, y_train, y_test = clean.train_test()
print(len(X))
print(len(X_train), len(X_test))
assert len(X) == (len(X_train)+len(X_test)), 'Not the correct length'

  # Modeling using a Gaussian Process Classifier from reduced dimensions
  # resulting from the above Principal Component Analysis
  def Gaussian_Process(self):
    # Want to take the n most relevant features from PCA & 