# HW4 Statistical Learning 760

# Gaussian Mixture Methods Classifier for digits
# Used two distributions for analysis, but could do 3, 4, etc...


##################
# Gaussian Mixture Model
##################
#%%
import numpy as np
import pandas as pd
from scipy import stats
from math import pi, sqrt, exp
from sklearn.mixture import GaussianMixture
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
import matplotlib.pyplot as plt
import seaborn as sns

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


####################################
# Printing example Gaussian

# Creating x input example values
x = np.linspace(start=-10,stop=10,num=1000)
# Creating the y-values from a normal pdf with mean of 0 and variance of 1.5
y = stats.norm.pdf(x, loc=0, scale=1.5)
# Plotting the distribution
#plt.plot(x,y)
####################################


####################################
# Example pdf creation
# class Gaussian():

#   def __init__(self,mu,sigma):
#     self.mu = mu
#     self.sigma = sigma
  
#   # Probability Density Function
#   def pdf(self,datum):
#     u = (datum - self.mu) / abs(self.sigma)
#     y = (1 / (sqrt(2*pi)*abs(self.sigma))) * exp(-u*u/2)
#     return y

# #gaussian of best fit
# data = X_0
# best_single = Gaussian(np.mean(data),np.std(data))
# print('Parameters, mean={0}, std={1}'.format(best_single.mu, best_single.sigma))
####################################



###############################
# Actual Code Start
###############################

import sklearn.datasets as datasets
import numpy as np
import matplotlib.pyplot as plt
import random
import operator



def em_algo_go(x, regulariation_parameter, number_of_gaussians_distributions, number_of_iterations):

    # Regularization Matrix with lambda across the diagonal and zero's elsewhere.
    regularization_term = np.multiply(regulariation_parameter, np.eye(x.shape[1]))
    sample_cov_mat = np.cov(x, rowvar=False) + regularization_term

    # mean = random sample point, covariance = sample covariance
    gaussian_models = []
    for i in range(number_of_gaussians_distributions):
        rand_mean = random.randint(0, x.shape[0])
        gaussian_models.append({"mean": x[rand_mean], "cov": sample_cov_mat, "gamma": 0.5})

    # x_responsibilities = [[Xi], [Gaussian_model_index]]
    x_responsibilities = []
    for i in range(x.shape[0]):
        buffer = [x[i], -1]
        x_responsibilities.append(buffer)

    # main loop
    for i in range(number_of_iterations):
        print("iteration: ", i)
        # expectation
        EStep(gaussian_models, x_responsibilities)

        # maximization
        MStep(gaussian_models, x_responsibilities, regularization_term)

    return gaussian_models


def EStep(dictionary_of_gaussian, data):
    """
    Assign a Gaussian to each data point
    :param dictionary_of_gaussian: dictionary of Gaussian models
    :param data: input data
    :return: modifies data directly
    """

    # pre-compute inverses of the covariance matrices because it's very expensive
    inverse_cov = []
    for model in dictionary_of_gaussian:
        inverse_cov.append(np.linalg.inv(model["cov"]))

    for i in range(len(data)):
        x_i = data[i][0]

        mahalanobis_distances = {}
        for j in range(len(dictionary_of_gaussian)):
            centered_point = x_i - dictionary_of_gaussian[j]["mean"]
            mahalanobis_distance = centered_point.dot(inverse_cov[j]).dot(centered_point)
            mahalanobis_distances[j] = mahalanobis_distance

        data[i][1] = min(mahalanobis_distances.items(), key=operator.itemgetter(1))[0]

    return


def MStep(dictionary_of_gaussian, data, regularization_term):
    """
    :param dictionary_of_gaussian: dictionary of Gaussian Models
    :param data: input data
    :param regularization_term: square matrix added to the covariance
    :return: dictionary_of_gaussian updated
    """
    size_of_data = len(data)
    for i in range(len(dictionary_of_gaussian)):
        x_cluster = np.array([t[0] for t in data if t[1] == i])

        dictionary_of_gaussian[i]["mean"] = x_cluster.mean(axis=0)
        dictionary_of_gaussian[i]["cov"] = np.cov(x_cluster, rowvar=False) + regularization_term
        dictionary_of_gaussian[i]["gamma"] = x_cluster.shape[0] / size_of_data

    return


####################################
# Generating Example

# test data generation code from: http://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html
# np.random.seed(0)

# n_samples = 500

# random_state = 170
# X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

# #%%

# gaussian_models = em_algo_go(X, 0.1, 3, 500)

# print("gaussian models: \n", gaussian_models)

# for i in range(len(gaussian_models)):
#     model = gaussian_models[i]
#     plt.scatter(model["mean"][0], model["mean"][1], color='r', s=10)
# plt.scatter(X[:, 0], X[:, 1], facecolors='none', linewidths=0.5, edgecolors='b', s=10)

# plt.show()
####################################



class Gauss_Clf_Abs:
    """
    Abstract Gaussian classifier providing an interface and common methods
    """

    def __init__(self, regulariation_parameter):
        self._regularization_param = regulariation_parameter
        self._class_models = {}
        return

    def additional_class(self, label, data_file):
        raise NotImplementedError("Method not implemented.")
        pass

    def classify(self, new_data_point):
        raise NotImplementedError("Method not implemented.")
        pass

    def _load_training_data(self, file_name):
        features = []

        with open(file_name, 'r') as file:
            for row in file:
                data_string = row.strip().split(',')
                data = []
                for i in range(len(data_string)):
                    data.append(float(data_string[i]))

                features.append(data)

        return np.array(features)

    def _calc_probability(self, covariance, centered_point):
        """
        :param covariance: covariance matrix
        :param centered_point: new point centered (x - mean)
        :return: probability
        """
        denominator_1 = ((2 * np.pi) ** (centered_point.shape[0] / 2))
        denominator_2 = np.sqrt(np.linalg.det(covariance))
        denominator = denominator_1 * denominator_2

        exp_term = centered_point.dot(np.linalg.inv(covariance)).dot(centered_point) * (-0.5)
        exp_val = np.exp(exp_term)

        return (1 / denominator) * exp_val





class gauss(Gauss_Clf_Abs):
    """
    Multi-class classifier that fits a Gaussian distribution to multidimensional data
    Note: assumes Gaussian distribution
    """

    def additional_class(self, label, data_file):
        # load training data
        matrix_training_data = super()._load_training_data(data_file)

        # find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = matrix_training_data.mean(axis=0)

        # find the covariance; row var=False because each column represents a variable, with rows as observations
        regularization_term = np.multiply(self._regularization_param, np.eye(matrix_training_data.shape[1]))
        cov_mat = np.cov(matrix_training_data, rowvar=False) + regularization_term

        # add to a dictionary
        self._class_models[label] = {'mean': mean_vec, 'cov': cov_mat}

        return self

    def classify(self, new_data_point):
        if not self._class_models:
            raise RuntimeError("no class models found")

        class_probabilities = {}

        for label, model in self._class_models.items():
            p_x = super()._calc_probability(model['cov'], new_data_point - model['mean'])
            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]



class pca(Gauss_Clf_Abs):
    """
    Multi-class classifier that fits a Gaussian distribution to multidimensional data
    with pca dimensionality reduction
    Note: assumes Gaussian distribution
    """

    def __init__(self, regulariation_parameter, num_principle_components):
        super().__init__(regulariation_parameter)
        self.num_components = num_principle_components
        return

    def additional_class(self, label, data_file):
        # Load training data
        matrix_training_data = super()._load_training_data(data_file)

        # Find the mean; axis 0 means mean of each column which represents observations of the feature
        mean_vec = matrix_training_data.mean(axis=0)

        # Find the covariance; row var=False because each column represents a variable, with rows as observations
        cov_mat = np.cov(matrix_training_data, rowvar=False)

        # Find and sort Eigenvalues/Eigenvectors
        eigen_vals, eigen_vecs = np.linalg.eig(cov_mat)
        eigen = list(zip(eigen_vals, eigen_vecs))
        n_principle_components = sorted(eigen, key=operator.itemgetter(0), reverse=True)[:self.num_components]

        # num_components x 256
        eigen_vec_mat = np.array([vec[1] for vec in n_principle_components])

        # n x 256
        training_data_centered = matrix_training_data - mean_vec

        # [n x 256] x [256 x num_components] = [n x num_components]
        projected_data = training_data_centered.dot(eigen_vec_mat.T)

        regularization_term = np.multiply(self._regularization_param, np.eye(self.num_components))
        projected_cov = np.cov(projected_data, rowvar=False) + regularization_term

        self._class_models[label] = {'mean': mean_vec, 'cov': projected_cov, 'eigen': eigen_vec_mat}

        return self

    def classify(self, new_data_point):
        class_probabilities = {}

        for label, model in self._class_models.items():
            centered_point = new_data_point - model['mean']
            projected_point = centered_point.dot(model['eigen'].T)
            p_x = super()._calc_probability(model['cov'], projected_point)
            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]





class mgc(Gauss_Clf_Abs):
    """
    Multi-class Mixture of Gaussians classifier
    """

    def __init__(self, regulariation_parameter=0.1, num_of_gaussian=2, em_algorithm_iterations=500):
        super().__init__(regulariation_parameter)
        self._number_of_gaussians_distributions = num_of_gaussian
        self._em_number_of_iterations = em_algorithm_iterations
        return

    def additional_class(self, label, data_file):
        # load training data
        matrix_training_data = super()._load_training_data(data_file)

        # call em algorithm and store the mean, covariance, and gamma
        gaussian_models = em_algo_go(matrix_training_data,
                                                              self._regularization_param,
                                                              self._number_of_gaussians_distributions,
                                                              self._em_number_of_iterations)

        self._class_models[label] = gaussian_models

        return self

    def classify(self, new_data_point):
        if not self._class_models:
            raise RuntimeError("no class models found")

        class_probabilities = {}
        for label, class_gaussian_models in self._class_models.items():
            p_x = 0
            for model in class_gaussian_models:
                p_x += super()._calc_probability(model['cov'], new_data_point - model['mean']) * model["gamma"]

            class_probabilities[label] = p_x

        return max(class_probabilities.items(), key=operator.itemgetter(1))[0]

training_data_files = {0: "/Users/kevin/Desktop/Stats 760/Homework1/train.0.txt",
                       1: "/Users/kevin/Desktop/Stats 760/Homework1/train.1.txt",
                       2: "/Users/kevin/Desktop/Stats 760/Homework1/train.2.txt",
                       3: "/Users/kevin/Desktop/Stats 760/Homework1/train.3.txt",
                       4: "/Users/kevin/Desktop/Stats 760/Homework1/train.4.txt",
                       5: "/Users/kevin/Desktop/Stats 760/Homework1/train.5.txt",
                       6: "/Users/kevin/Desktop/Stats 760/Homework1/train.6.txt",
                       7: "/Users/kevin/Desktop/Stats 760/Homework1/train.7.txt",
                       8: "/Users/kevin/Desktop/Stats 760/Homework1/train.8.txt",
                       9: "/Users/kevin/Desktop/Stats 760/Homework1/train.9.txt"}

out_directory = "/Users/kevin/Desktop/"


def load_test_data(file_name):
    test_data_buffer = []
    with open(file_name, 'r') as file:
        for row in file:
            data_string = row.strip().split()
            data = []
            class_label = -1
            for i in range(len(data_string)):
                if i == 0:
                    class_label = int(data_string[i])
                    continue
                data.append(float(data_string[i]))

            test_data_buffer.append((class_label, np.array(data)))

    return test_data_buffer


def run_test_data(test_data_set, gauss_classifier):
    # row indexed by the test label, column indexed by predicted class
    confusion_matrix = np.zeros((10, 10))

    counter = 0
    for p in test_data_set:
        predicted_class = gauss_classifier.classify(p[1])
        confusion_matrix[p[0]][predicted_class] += 1
        counter += 1
        print("count: ", counter)

    return confusion_matrix


def calc_performance_from_confusion(confusion_mat):
    total_test_data = np.sum(confusion_mat)

    # calculate the accuracy and error rate (1 - accuracy)
    accuracy = np.round(np.trace(confusion_mat) / total_test_data, 5)
    error_rate = np.round(1.0 - accuracy, 5)

    # calculate the precision of each class (index of array representing the digit)
    precisions = np.zeros((10, 1))
    for i in range(0, 10):
        precisions[i] = np.round(confusion_mat[i][i] / np.sum(confusion_mat[i]), 5)

    return accuracy, error_rate, precisions


def save_data_to_csv(data, file_name, csv_header, data_format):
    np.savetxt(file_name,
               data,
               delimiter=",",
               header=csv_header,
               fmt=data_format)
    return


def run_default_classifier(test_data_set):
    classifier = gauss(0.1)

    for label, file in training_data_files.items():
        classifier.additional_class(label, file)

    confusion = run_test_data(test_data_set, classifier)
    accuracy, error, precisions = calc_performance_from_confusion(confusion)

    csv_file_name = out_directory + "all_features_metrics.csv"
    csv_file_header = "Accuracy,Error"

    for i in range(0, 10):
        csv_file_header += ",Precision of {}".format(i)

    save_data_to_csv(np.insert(precisions, 0, (accuracy, error)),
                     csv_file_name,
                     csv_file_header,
                     '%.5f')

    print("done!! \n", confusion)

    return


def visualize_pca_run(run_results):
    plt.scatter(run_results[:, 0], run_results[:, 2], facecolors='none', linewidths=0.5, edgecolors='b', s=10)
    plt.plot(run_results[:, 0], run_results[:, 2], 'r-', linewidth=0.5)

    x1, x2, y1, y2 = plt.axis()
    plt.axis((x1, x2, 0, 1))

    plt.title("Classifier Error with Varying Dimensions")
    plt.xlabel("Number of Dimensions")
    plt.ylabel("Error Rate")

    plt.show()

    return


def run_pca_classifier(test_data_set):
    results = []
    for num_components in range(5, 155, 5):
        classifier = pca(0.1, num_components)

        for label, file in training_data_files.items():
            classifier.additional_class(label, file)

        confusion = run_test_data(test_data_set, classifier)
        accuracy, error, precisions = calc_performance_from_confusion(confusion)
        results.append((num_components, accuracy, error))
        print("num_component completed: ", num_components)

    csv_file_name = out_directory + "pca_metrics.csv"
    csv_file_header = "Number of Dimensions,Accuracy,Error"
    save_data_to_csv(results,
                     csv_file_name,
                     csv_file_header,
                     '%.5f')

    visualize_pca_run(np.array(results))

    return


def run_mixture_of_gaussian(test_data_set):
    number_of_gaussians_distributions = 2
    classifier = mgc(num_of_gaussian=number_of_gaussians_distributions, em_algorithm_iterations=5)

    for label, file in training_data_files.items():
        classifier.additional_class(label, file)

    confusion = run_test_data(test_data_set, classifier)

    confusion_file_prefix = "/Users/kevin/Desktop/"
    output_file_name = confusion_file_prefix + "mgc_confusion_" + str(number_of_gaussians_distributions) + "_models.csv"
    np.savetxt(output_file_name,
               confusion,
               fmt='%i',
               delimiter=',')

    print("done!! \n", confusion)

    return

test_data = load_test_data("/Users/kevin/Desktop/Stats 760/Homework1/test.txt")
run_mixture_of_gaussian(test_data)