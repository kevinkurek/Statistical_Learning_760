# HW3 Statistical Learning 760

# Principal Component Analysis fed into a
# Gaussian Process Classifier for digits

#%%
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt

df_0 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.0.txt',header=None)
df_0['Digit'] = 0
df_1 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.1.txt',header=None)
df_1['Digit'] = 1
df_2 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.2.txt',header=None)
df_2['Digit'] = 2
df_3 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.3.txt',header=None)
df_3['Digit'] = 3
df_4 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.4.txt',header=None)
df_4['Digit'] = 4
df_5 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.5.txt',header=None)
df_5['Digit'] = 5
df_6 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.6.txt',header=None)
df_6['Digit'] = 6
df_7 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.7.txt',header=None)
df_7['Digit'] = 7
df_8 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.8.txt',header=None)
df_8['Digit'] = 8
df_9 = pd.read_csv('/Users/kevin/Desktop/Stats 760/Homework1/train.9.txt',header=None)
df_9['Digit'] = 9

#######################################################

class gaussian_process():

  def __init__(self,df_list):
    self.df_list = df_list

  def clean_df(self):
    """ Input: list of dataframes coming from __init__ method
        Intermediate: Principal Component Analysis to determine top
                      eigenvalues to use for transforming and fitting
                      the data, rather than using all 256 input features.
                      I subjectively chose 20 as the variance beyond 20
                      principal components didn't seem significant. Graphed
                      for easier visualization.
        Output: Scaled (data centered) list of X input dataframes &
                list of y output features."""

    self.scaled_X = []
    self.scaled_y = []

    for df in self.df_list:

      # Cleaning, scaling, fitting, & transforming original data
      # to eigenvalues resulting in PCA components
      self.X = scale(df.iloc[:,:-1])
      self.pca = PCA(n_components=20)
      self.pca.fit(self.X)
      self.X = self.pca.fit_transform(self.X) # Transforming original matrix, X, to PCA components.
      self.new_df = pd.DataFrame(self.X)
      self.scaled_X.append(self.new_df)

      self.y = df.iloc[:,-1]
      self.scaled_y.append(self.y)


    # Graphically showing PCA feature number (sorted highest eigenvalue number to lowest)
    # by explained variance within those features.
    self.features = range(self.pca.n_components_)
    plt.bar(self.features, self.pca.explained_variance_)
    plt.xlabel('PCA feature')
    plt.ylabel('variance')
    plt.xticks(self.features)
    plt.show()

    return 'Finished Cleaning'
  
  def concat_df(self):
    """ Input: scaled_X which is a list of input dataframes
                scaled_y which is a list of the class vectors (i.e. if
                the row corresponds to 0, 1, 2, etc...)
        Output: Completely concatenated input feature array & completely
                concatenated classification array"""

    self.df_X_together = pd.concat(self.scaled_X)
    self.df_y_together = pd.concat(self.scaled_y)

    self.X_input_array = np.array(self.df_X_together)
    self.y_input_array = np.array(self.df_y_together)
    
    return 'Finished Concatenating'

  
  def gaussian_model(self):
    """ Input: X_input_array, y_input_array from above method
        Output: Gaussian Classifier with Classification Report
                containing precision, recall, & f1 score of predictions"""

    # train_test_split on X_input_array and y_input_array
    self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X_input_array, self.y_input_array, test_size=.3)

    Gaussian = GaussianProcessClassifier()
    Gaussian.fit(self.X_train,self.y_train)
    self.ypred = Gaussian.predict(self.X_test)
    #print(classification_report(self.y_test,self.ypred))
    print(confusion_matrix(self.y_test,self.ypred))
    return 'Done'

# Call Instance using digits dataframes in a list
digits = gaussian_process([df_0,df_1,df_2,df_3,df_4,df_5,df_6,df_7,df_8,df_9])
digits.clean_df()
digits.concat_df()
digits.gaussian_model()