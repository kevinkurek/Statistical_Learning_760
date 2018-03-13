# HW6 Statistical Learning 760

# PCA to Neural Network from Scratch

###########################
# Steps for PCA

# 1. Scale Data
# 2. Get sorted eigenvalues, take top 16. (The number 16 was told to us)
# 3. Take 16 values and pipeline into NN

# Steps for NN

# 1. Feed Forward
# 2. Calculate Error
# 3. Backpropigation to adjust weights
# 4. Repeat
###########################

# Data
# Digits dataset with all 10 values
#%%
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt


def read_in_pca_transform(path):

  """ Input: Path to digit txt files
      Output: A list of dataframes that have been scaled (centered) and 
              PCA transformed for the top 16 components within the digits datasets. """

  # Reading and Concatenating Data into single dataframe
  df_list = []
  for i in range(10):
    new_file = pd.read_csv(path+str(i)+'.txt', header=None)

    # PCA analysis with 16 components before concatentions
    new_file = scale(new_file) # Centers Data
    pca = PCA(n_components=16)
    pca.fit(new_file)
    new_file = pca.fit_transform(new_file)

    # Making DF before adding additional y column of target "Digit"
    new_file = pd.DataFrame(new_file)

    # Adding y target variable "Digit" to specify which rows correspond to which digit
    new_file['Digit'] = i

    # Appending individual dataframes to a list
    df_list.append(new_file)

  return df_list

path = '/Users/kevin/Desktop/Stats 760/Homework1/train.'
df_list = read_in_pca_transform(path)


# Concatenating all dataframes in df_list together
training_df = pd.concat(df_list)
print(training_df.shape)
training_df.tail()




#%%
################
# Start of NN from scratch
# Can interchange the activation function easily
################

training_size = training_df.shape[0] # Size of training set
number_of_input_features = training_df.shape[1] - 1 # Number of Input Features; Subtract 1 for y target variable
neurons_in_hidden_layer = 16 # neurons in hidden layer
neurons_in_output_layer = 10
alpha = 0.00001 # Learning Rate
reg_lambda = 0.01 # Regularization Term

X_train, X_test, y_train, y_test = train_test_split(training_df.iloc[:,:-1],training_df.iloc[:,-1],test_size=0.3)

X = np.array(X_train, dtype=np.float128)
y = np.array(y_train)
y = y.reshape((-1,1))
num_examples = len(X)


# Define Sigmoid Activation Function
def sigmoid(x):
  return 1/(1+np.exp(x))
# Testing, sigmoid(0) = 0.5


# Define Loss function calculation
def calc_loss(model,activation):
  W1, b1, W2, b2 = model['W1'],model['b1'],model['W2'],model['b2']

  # Forward Propagation
  z1 = X.dot(W1) + b1
  a1 = activation(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2) # Numerator for softmax
  probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True) # Softmax

  # Calculate Loss (categorical cross entropy)
  neg_log = -1*np.log(probs[range(num_examples),y]) 
  data_loss = np.sum(neg_log)
  data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2))) #Regularized model to reduce loss
  loss = 1/num_examples * data_loss

  return loss


# Define prediction
def predict(model,x,activation):

  # Forward Propagation
  z1 = X.dot(W1) + b1
  a1 = activation(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2) # Numerator for softmax
  probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True) # Softmax

  # Prediction
  y_hat = np.argmax(probs,axis=1)

  return y_hat


def build_model(neurons_in_hidden_layer,activation,num_passes=100,print_loss=False):

  # Random Initialization of Weights/Bias
  #np.random.seed(0)
  W1 = np.random.randn(number_of_input_features,neurons_in_hidden_layer) / np.sqrt(number_of_input_features)
  b1 = np.zeros((len(X),number_of_input_features))
  W2 = np.random.randn(neurons_in_hidden_layer,neurons_in_output_layer) / np.sqrt(neurons_in_hidden_layer)
  b2 = np.zeros((len(X),neurons_in_output_layer))

  # Initialize our model
  model = {}

  # Gradient Descent for each batch
  for i in range(num_passes):

    # Forward Propagation
    z1 = X.dot(W1) + b1
    a1 = activation(z1)
    z2 = a1.dot(W2) + b2
    exp_scores = np.exp(z2)
    probs = exp_scores / np.sum(exp_scores,axis=1,keepdims=True)
    #print(probs)

    # Backpropagation
    delta3 = probs
    delta3[range(num_examples),y] -= 1 ###############
    dW2 = (a1.T).dot(delta3)
    db2 = np.sum(delta3, axis=0, keepdims=True)
    delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
    dW1 = np.dot(X.T, delta2)
    db1 = np.sum(delta2, axis=0)

    # Add regularization terms
    dW2 += reg_lambda * W2
    dW1 += reg_lambda * W1

    # Gradient Descent parameter updates
    W1 += -alpha * dW1
    b1 += -alpha * db1
    W2 += -alpha * dW2
    b2 += -alpha * db2

    model = {'W1':W1, 'b1':b1, 'W2':W2, 'b2':b2}

    if print_loss and i % 20 == 0:
      print("Loss after iteration %i: %f" % (i, calc_loss(model,activation)))
    
  return model


# Build a model with a 3-dimensional hidden layer
model = build_model(neurons_in_hidden_layer, activation=sigmoid, print_loss=True)

# Plot the decision boundary
# plot_decision_boundary(lambda x: predict(model, x))
# plt.title("Decision Boundary for hidden layer size 3")
















##################################################
# Keras Model
##################################################

#%%
import pandas as pd
import numpy as np
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
#from sklearn.cross_validation import cross_val_predict #Depricated
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras import backend

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(training_df.iloc[:,:-1], training_df.iloc[:,-1], test_size=0.2)


# Create Baseline model
seed = 7

def baseline_model():
  model = Sequential()
  model.add(Dense(14, input_dim=16, kernel_initializer='normal', activation='relu'))
  model.add(Dense(12,activation='relu'))
  model.add(Dense(10, kernel_initializer='normal', activation='softmax'))

  # Compile model
  model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

# Standardize and Pipeline
estimators=[]
estimators.append(('standardize',StandardScaler()))
estimators.append(('mlp', KerasClassifier(build_fn=baseline_model, epochs=100, batch_size=5,verbose=0)))
pipeline=Pipeline(estimators)

# K-Fold Cross-Validation & Cross-Validation Score
# kfold = KFold(n_splits=3, shuffle=True, random_state=seed)
# results = cross_val_score(pipeline, X, y, cv=kfold)
# print(results)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_test)
cm = confusion_matrix(y_test, y_pred)
cm = pd.DataFrame(cm)
print(classification_report(y_test, y_pred))

#%%
cm
























############################################
# Test Data Set
############################################

#%%
import numpy as np
from sklearn import datasets

# Generate a dataset and plot it for testing
#np.random.seed(0)
X, y = datasets.make_moons(200, noise=0.20)
plt.scatter(X[:,0], X[:,1], s=40, c=y, cmap=plt.cm.Spectral)
print(y.shape,y)
#%%


num_examples = len(X) # training set size
nn_input_dim = 2 # input layer dimensionality
nn_output_dim = 2 # output layer dimensionality
 
# Gradient descent parameters (I picked these by hand)
epsilon = 0.01 # learning rate for gradient descent
reg_lambda = 0.01 # regularization strength


# Helper function to evaluate the total loss on the dataset
def calculate_loss(model):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  # Forward propagation to calculate our predictions
  z1 = X.dot(W1) + b1
  a1 = np.tanh(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  # Calculating the loss
  corect_logprobs = -np.log(probs[range(num_examples), y])
  data_loss = np.sum(corect_logprobs)
  # Add regulatization term to loss (optional)
  data_loss += reg_lambda/2 * (np.sum(np.square(W1)) + np.sum(np.square(W2)))
  return 1./num_examples * data_loss


# Helper function to predict an output (0 or 1)
def predict(model, x):
  W1, b1, W2, b2 = model['W1'], model['b1'], model['W2'], model['b2']
  # Forward propagation
  z1 = x.dot(W1) + b1
  a1 = np.tanh(z1)
  z2 = a1.dot(W2) + b2
  exp_scores = np.exp(z2)
  probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
  return np.argmax(probs, axis=1)


# This function learns parameters for the neural network and returns the model.
# - nn_hdim: Number of nodes in the hidden layer
# - num_passes: Number of passes through the training data for gradient descent
# - print_loss: If True, print the loss every 1000 iterations
def build_model(nn_hdim, num_passes=20000, print_loss=False):
     
  # Initialize the parameters to random values. We need to learn these.
  np.random.seed(0)
  W1 = np.random.randn(nn_input_dim, nn_hdim) / np.sqrt(nn_input_dim)
  b1 = np.zeros((1, nn_hdim))
  W2 = np.random.randn(nn_hdim, nn_output_dim) / np.sqrt(nn_hdim)
  b2 = np.zeros((1, nn_output_dim))

  # This is what we return at the end
  model = {}
    
  # Gradient descent. For each batch...
  for i in range(0, num_passes):

      # Forward propagation
      z1 = X.dot(W1) + b1
      a1 = np.tanh(z1)
      z2 = a1.dot(W2) + b2
      exp_scores = np.exp(z2)
      probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

      # Backpropagation
      delta3 = probs
      delta3[range(num_examples), y] -= 1
      dW2 = (a1.T).dot(delta3)
      db2 = np.sum(delta3, axis=0, keepdims=True)
      delta2 = delta3.dot(W2.T) * (1 - np.power(a1, 2))
      dW1 = np.dot(X.T, delta2)
      db1 = np.sum(delta2, axis=0)

      # Add regularization terms (b1 and b2 don't have regularization terms)
      dW2 += reg_lambda * W2
      dW1 += reg_lambda * W1

      # Gradient descent parameter update
      W1 += -epsilon * dW1
      b1 += -epsilon * db1
      W2 += -epsilon * dW2
      b2 += -epsilon * db2
        
      # Assign new parameters to the model
      model = { 'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
        
      # Optionally print the loss.
      # This is expensive because it uses the whole dataset, so we don't want to do it too often.
      if print_loss and i % 1000 == 0:
        print("Loss after iteration %i: %f" %(i, calculate_loss(model)))
    
  return model


# Build a model with a 3-dimensional hidden layer
model = build_model(3, print_loss=True)



