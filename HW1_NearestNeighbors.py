# HW1 Statistical Learning 760

# Nearest Neighbors Classifier for digits

#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
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

#%%

class knn_algo():

  def __init__(self,df_list):
    self.df_list = df_list

  # Concatentate digit 0 and digit 3 dataframes together
  def concat_dfs(self):
    self.df = pd.concat(self.df_list)
    print(len(self.df))
    return self.df

  # Subsetting to input vector X, and output value y.
  def features(self):
    self.X = self.df.iloc[:,:-1]
    self.y = self.df.iloc[:,-1]
    print(len(X.columns)) # Make sure there are 256 input features
    return self.X,self.y

  def knn_model(self):

    # Range of number of neighbors to iterate through
    number_of_neighbors = list(range(1,10))

    # Empty list to attach scores to
    cv_scores = []

    for i in number_of_neighbors:
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y, test_size=.3, random_state=7,stratify=self.y)
      self.knn = KNeighborsClassifier(n_neighbors=i)
      scores = cross_val_score(self.knn,self.X_train,self.y_train,cv=10,scoring='accuracy') # 10-fold Cross Validation
      cv_scores.append(scores.mean())
    print(cv_scores)

    index, value = max(enumerate(cv_scores), key=operator.itemgetter(1))
    print('The maximum accuracy is: {}'.format(value))
    print('The optimal number of neighbors is: {}'.format(index+1))
    # print(confusion_matrix(self.y_test, self.y_pred))
    # print(classification_report(self.y_test,self.y_pred))
    return 'It worked'

first = knn_algo([df_0,df_3])
first.concat_dfs()
first.features()
print(first.knn_model())

second = knn_algo([df_2,df_7])
second.concat_dfs()
second.features()
print(second.knn_model())

third = knn_algo([df_1,df_9])
third.concat_dfs()
third.features()
print(third.knn_model())