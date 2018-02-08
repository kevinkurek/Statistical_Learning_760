# HW2 part 2 Statistical Learning 760

# Calculating Residual Sum of Squares for k-subsets of
# Prostate Cancer Data

# Prostate cancer k-subset optimization for linear Regression

#%%
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
from itertools import combinations
from sklearn.feature_selection import RFE
from sklearn.metrics import mean_squared_error
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# Read in txt data
df = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework2/prostate.txt',sep='\t') #tab delimited
df = df.drop(df.columns[0],axis=1) # First column was a duplicate index so I dropped it
df.head()

# Splitting into same training and test set used in the data
training_df = df[df['train']=='T']
test_df = df[df['train']=='F']
training_df = training_df.drop('train', axis=1)
test_df = test_df.drop('train', axis=1)
assert len(df)==(len(training_df)+len(test_df)), 'Lengths dont match original data' # Didn't show, so they match


# Split training_df into X and y components
X_train_original = training_df.iloc[:,:-1]
X_test_original = test_df.iloc[:,:-1]
y_train_original = training_df.iloc[:,-1]
y_test_original = test_df.iloc[:,-1]

# Converting originals to arrays (sk-learn tends to like arrays better)
X_train = np.array(training_df.iloc[:,:-1])
y_train = np.array(training_df.iloc[:,-1])
X_test = np.array(test_df.iloc[:,:-1])
y_test = np.array(test_df.iloc[:,-1])

# X_train_feature1 = X_train[:,0].reshape(-1,1)
# X_test_feature1 = X_test[:,0].reshape(-1,1)
# print(X_test_feature1)
# print(y_train)
X_train_original
X_test_original
y_train_original
y_test_original

# Create a class of above functions

class k_subset_regression():

  def __init__(self,input_features_df,choose_k):
    self.input_features_df = input_features_df
    self.choose_k = choose_k

  def combin(self):
    """ Input: Takes a list of columns, L, and the number of columns, k, you choose for your subset
        returns: subset combinations of L choose k """
    combo = list(combinations(self.input_features_df,self.choose_k))

    self.newlist = []
    for i in combo:
      self.newlist.append(i)
    return self.newlist

  # combination_list = combin(X_train_original,2)

  def subset_dataframes(self):
    """ Input: train or test data you wish to create subsets of
        returns: a list of all the subsets of combinations you specified in combin() """

    list_of_subset_df = [] 
    for column in self.newlist:
      list_column = list(column)
      #print(list_column)
      subset_df = pd.DataFrame(self.input_features_df[list_column])
      list_of_subset_df.append(subset_df)
    return list_of_subset_df

k_subset_reg = k_subset_regression(X_train_original,1)
k_subset_reg.combin()
list_of_subset_df_train1 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,1)
k_subset_reg.combin()
list_of_subset_df_test1 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,2)
k_subset_reg.combin()
list_of_subset_df_train2 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,2)
k_subset_reg.combin()
list_of_subset_df_test2 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,3)
k_subset_reg.combin()
list_of_subset_df_train3 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,3)
k_subset_reg.combin()
list_of_subset_df_test3 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,4)
k_subset_reg.combin()
list_of_subset_df_train4 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,4)
k_subset_reg.combin()
list_of_subset_df_test4 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,5)
k_subset_reg.combin()
list_of_subset_df_train5 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,5)
k_subset_reg.combin()
list_of_subset_df_test5 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,6)
k_subset_reg.combin()
list_of_subset_df_train6 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,6)
k_subset_reg.combin()
list_of_subset_df_test6 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,7)
k_subset_reg.combin()
list_of_subset_df_train7 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,7)
k_subset_reg.combin()
list_of_subset_df_test7 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_train_original,8)
k_subset_reg.combin()
list_of_subset_df_train8 = k_subset_reg.subset_dataframes()

k_subset_reg = k_subset_regression(X_test_original,8)
k_subset_reg.combin()
list_of_subset_df_test8 = k_subset_reg.subset_dataframes()


def loop_regression(list_of_subset_df_train,y_train,list_of_subset_df_test,y_test):

  for train,test in zip(list_of_subset_df_train,list_of_subset_df_test):

    reg = linear_model.LinearRegression()
    reg.fit(train,y_train)
    y_pred = reg.predict(test)
    rss = ((y_pred - y_test)**2).sum() # Residual Sum of Squares
    r2_auto = reg.score(test,y_test)

  return rss

rss_1 = loop_regression(list_of_subset_df_train1, y_train_original, list_of_subset_df_test1, y_test_original)
rss_2 = loop_regression(list_of_subset_df_train2, y_train_original, list_of_subset_df_test2, y_test_original)
rss_3 = loop_regression(list_of_subset_df_train3, y_train_original, list_of_subset_df_test3, y_test_original)
rss_4 = loop_regression(list_of_subset_df_train4, y_train_original, list_of_subset_df_test4, y_test_original)
rss_5 = loop_regression(list_of_subset_df_train5, y_train_original, list_of_subset_df_test5, y_test_original)
rss_6 = loop_regression(list_of_subset_df_train6, y_train_original, list_of_subset_df_test6, y_test_original)
rss_7 = loop_regression(list_of_subset_df_train7, y_train_original, list_of_subset_df_test7, y_test_original)
rss_8 = loop_regression(list_of_subset_df_train8, y_train_original, list_of_subset_df_test8, y_test_original)

list_rss = [rss_1,rss_2,rss_3,rss_4,rss_5,rss_6,rss_7,rss_8]

plt.plot(range(1,9),list_rss)
plt.title('Sum of Square Residuals of k-subset selection using Linear Regression')
plt.xlabel('Subset Size k')
plt.ylabel('Residual Sum of Squares')
plt.show()














#%%
# Manually designed single Linear Regression Algorithm to check
X = X_train_feature1
y = y_train

# Calculating means
X_mean = np.mean(X)
y_mean = np.mean(y)


# Getting Coefficients for beta1 & beta0
m = len(X)
numerator = 0
denominator = 0
for i in range(m):
  numerator += ((X[i]-X_mean)*(y[i]-y_mean)) # (X value - mean of X)*(y value - mean of y)
  denominator += (X[i]-X_mean)**2 # (X value - mean of X) squared

beta1 = numerator/denominator
beta0 = y_mean - beta1 * X_mean

# Fitting to Test Data, predicting, and calculating RSS value manually
m = len(X_test_feature1)
rss_manual = 0
for i in range(m):
  y_pred = beta0 + beta1 * X_test_feature1[i]
  rss_manual += ((y_test[i] - y_pred)**2).sum()
print(rss_manual)
# RSS is 14.39

# Calculating R^2 value
ss_t = 0
ss_r = 0
for i in range(m):
  y_pred = beta0 + beta1 * X_test_feature1[i]
  ss_t += (y_test[i] - y_mean) ** 2
  ss_r += (y_test[i] - y_pred) ** 2
r2_manual = 1 - (ss_r/ss_t)
print(r2_manual)
# R2 is .5460, which isn't good, but that's why we don't use a single input feature
# in order to predict the output feature.

#%%
# Scikit-Learn Approach for Linear Regression
reg = linear_model.LinearRegression()
reg.fit(X_train_feature1,y_train)
y_pred = reg.predict(X_test_feature1)
rss_automatic = ((y_pred - y_test)**2).sum()
r2_auto = reg.score(X_test_feature1,y_test)
print(rss_automatic)
print(r2_auto)
# RSS is 14.39, identical to manual
# R2 is .5429, which is almost identical to manual





# #%%
# class k_subset_regression():

#   def __init__(self,input_features_df,choose_k):
#     self.input_features_df = input_features_df
#     self.choose_k = choose_k

#   def combin(self):
#     """ Input: Takes a list of columns, L, and the number of columns, k, you choose for your subset
#         returns: subset combinations of L choose k """
#     combo = list(combinations(self.input_features_df,self.choose_k))

#     self.newlist = []
#     for i in combo:
#       self.newlist.append(i)
#     return self.newlist

#   # combination_list = combin(X_train_original,2)

#   def subset_dataframes(self):
#     """ Input: train or test data you wish to create subsets of
#         returns: a list of all the subsets of combinations you specified in combin() """

#     list_of_subset_df = [] 
#     for column in self.newlist:
#       list_column = list(column)
#       #print(list_column)
#       subset_df = pd.DataFrame(self.input_features_df[list_column])
#       list_of_subset_df.append(subset_df)
#     return self.list_of_subset_df

#   def iterate_list_of_subsets(list_of_subset_df_train,list_of_subset_df_test):


# #%%
# def iterate_list_of_subsets(X_train_original):

#   list_of_all_combinations_train = []
#   list_of_all_combinations_test = []

#   for i in range(len(X_train_original)):
#     k_subset_reg = k_subset_regression(X_train_original,i)
#     k_subset_reg.combin()
#     list_of_subset_df_train = k_subset_reg.subset_dataframes()
#     list_of_all_combinations_train.append(list_of_subset_df_train)
#     print(list_of_all_combinations_train)


#   for j in range(len(X_train_original)):
#     k_subset_reg = k_subset_regression(X_test_original,j)
#     k_subset_reg.combin()
#     list_of_subset_df_test = k_subset_reg.subset_dataframes()
#     list_of_all_combinations_test.append(list_of_subset_df_test)
  
#   return list_of_all_combinations_train, list_of_all_combinations_test

# iterate_list_of_subsets(X_train_original)
















#%%
# Obtaining rank from each input variable in the data set
# If they have a rank of 1 then that means they contribute to
# the model and should be kept

estimator = linear_model.LinearRegression()
selector = RFE(estimator, 8, step=1)
selector = selector.fit(X_train,y_train)
selector.ranking_


#%%
# SFS allows you to do k-subset selection in a single line of code
# the 8 tells it to loop through 8 features and return the negative mean squared error

reg = linear_model.LinearRegression()
sfs = SFS(reg,k_features=8,forward=True,floating=False,scoring='neg_mean_squared_error',cv=10)
sfs = sfs.fit(X_train,y_train)

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err')
plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()
