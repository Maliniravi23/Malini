#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 25 20:06:15 2021

@author: tenet
"""

# importing libraries
import math
import pandas as pd
import numpy as np
import scipy
import glob
from sklearn.cluster import KMeans
from scipy.stats import multivariate_normal as mvn


# function for gmm varaiable
class GMM:

    def __init__(self, C, n_runs):
        self.C = C  # number of Guassians/clusters
        self.n_runs = n_runs

    def get_params(self):
        return (self.mu, self.pi, self.sigma)

    # ===========================================================================#
    # def calculate the mean covaraince
    def calculate_mean_covariance(self, X, prediction):

        d = X.shape[1]
        labels = np.unique(prediction)
        self.initial_means = np.zeros((self.C, d))
        self.initial_cov = np.zeros((self.C, d, d))
        self.initial_pi = np.zeros(self.C)

        counter = 0
        for label in labels:
            ids = np.where(prediction == label)  # returns indices
            self.initial_pi[counter] = len(ids[0]) / X.shape[0]
            self.initial_means[counter, :] = np.mean(X[ids], axis=0)
            de_meaned = X[ids] - self.initial_means[counter, :]
            Nk = X[ids].shape[0]  # number of data points in current gaussian
            self.initial_cov[counter, :, :] = np.dot(self.initial_pi[counter] * de_meaned.T, de_meaned) / Nk
            counter += 1
        # assert np.sum(self.initial_pi) == 1

        return (self.initial_means, self.initial_cov, self.initial_pi)

    # ==================================================================================#
    # initialise the parameter
    def _initialise_parameters(self, X):

        n_clusters = self.C
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", max_iter=500, algorithm='auto')
        fitted = kmeans.fit(X)
        prediction = kmeans.predict(X)
        self._initial_means, self._initial_cov, self._initial_pi = self.calculate_mean_covariance(X, prediction)

        return (self._initial_means, self._initial_cov, self._initial_pi)

    # ========================================================================================#
    # initialise the expectation step
    def _e_step(self, X, pi, mu, sigma):

        N = X.shape[0]
        self.gamma = np.zeros((N, self.C))

        const_c = np.zeros(self.C)

        self.mu = self.mu if self._initial_means is None else self._initial_means
        self.pi = self.pi if self._initial_pi is None else self._initial_pi
        self.sigma = self.sigma if self._initial_cov is None else self._initial_cov

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule
            self.gamma[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        # normalize across columns to make a valid probability
        gamma_norm = np.sum(self.gamma, axis=1)[:, np.newaxis]
        self.gamma /= gamma_norm

        return self.gamma

    # ==============================================================================================#
    # initialise the maximization
    def _m_step(self, X, gamma):

        N = X.shape[0]  # number of objects
        C = self.gamma.shape[1]  # number of clusters
        d = X.shape[1]  # dimension of each object

        # responsibilities for each gaussian
        self.pi = np.mean(self.gamma, axis=0)

        self.mu = np.dot(self.gamma.T, X) / np.sum(self.gamma, axis=0)[:, np.newaxis]

        for c in range(C):
            x = X - self.mu[c, :]  # (N x d)

            gamma_diag = np.diag(self.gamma[:, c])
            x_mu = np.matrix(x)
            gamma_diag = np.matrix(gamma_diag)

            sigma_c = x.T * gamma_diag * x
            self.sigma[c, :, :] = (sigma_c) / np.sum(self.gamma, axis=0)[:, np.newaxis][c]

        return self.pi, self.mu, self.sigma

    # ========================================================================================#
    # iteration loss function
    def _compute_loss_function(self, X, pi, mu, sigma):

        N = X.shape[0]
        C = self.gamma.shape[1]
        self.loss = np.zeros((N, C))

        for c in range(C):
            dist = mvn(self.mu[c], self.sigma[c], allow_singular=True)
            self.loss[:, c] = self.gamma[:, c] * (
                        np.log(self.pi[c] + 0.00001) + dist.logpdf(X) - np.log(self.gamma[:, c] + 0.000001))
        self.loss = np.sum(self.loss)
        return self.loss

    # probability fit function
    def fit(self, X):

        d = X.shape[1]
        self.mu, self.sigma, self.pi = self._initialise_parameters(X)

        try:
            for run in range(self.n_runs):
                self.gamma = self._e_step(X, self.mu, self.pi, self.sigma)
                self.pi, self.mu, self.sigma = self._m_step(X, self.gamma)
                loss = self._compute_loss_function(X, self.pi, self.mu, self.sigma)

                if run % 10 == 0:
                    print("Iteration: %d Loss: %0.6f" % (run, loss))


        except Exception as e:
            print(e)

        return self

    # =============================================================================================#
    # prediction function
    def predict(self, X):

        labels = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            labels[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])
        labels = labels.argmax(1)
        return labels

    # ===============================================================================================#
    def predict_proba(self, X):

        post_proba = np.zeros((X.shape[0], self.C))

        for c in range(self.C):
            # Posterior Distribution using Bayes Rule, try and vectorise
            post_proba[:, c] = self.pi[c] * mvn.pdf(X, self.mu[c, :], self.sigma[c])

        return post_proba


# =============================================================================================#
# three basics parameter of gmm
# access all file mfcc extraction(200)

path = r'/home/tenet/Desktop/mfcc_ntimit'  # use your path

all_files = glob.glob(path + "/*.csv")
# split the data train and test 80% and 20%
train = []  # Train data
test = []  # test data
# ================================================================================================#
# import libray function
from sklearn.model_selection import train_test_split

for filename in all_files:
    df = pd.read_csv(filename, sep=' ', header=0)
    df = df.iloc[:, 1:].values
    x, y = train_test_split(df, train_size=0.8)
    train.append(x)
    test.append(y)

# ===================================================================================================#
# gmm basics parameter
cluster = []  # means
gmm_covar = []  # covarience
each_weights = []  # pi k


# ==================================================================================================#
def calculate_mean_covariance(train):
    global cluster, gmm_norm_weights, gmm_covar, weights
    for i in range(len(train)):


# calculate mean and centres
m = train.shape[0]
n = train.shape[1]
n_iter = 100
K =
Centroids = np.array([]).reshape(n, 0)
for i in range(K):
    rand = rd.randint(0, m - 1)
    Centroids = np.c_[Centroids, X[rand]]
Output = {}  # create dictionary

EuclidianDistance = np.array([]).reshape(m, 0)
for k in range(K):
    tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
    EuclidianDistance = np.c_[EuclidianDistance, tempDist]
C = np.argmin(EuclidianDistance, axis=1) + 1
Y = {}
for k in range(K):
    Y[k + 1] = np.array([]).reshape(19, 0)
for i in range(m):
    Y[C[i]] = np.c_[Y[C[i]], X[i]]

for k in range(K):
    Y[k + 1] = Y[k + 1].T

for k in range(K):
    Centroids[:, k] = np.mean(Y[k + 1], axis=0)
for i in range(n_iter):
    # step 2.a
    EuclidianDistance = np.array([]).reshape(m, 0)
    for k in range(K):
        tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
        EuclidianDistance = np.c_[EuclidianDistance, tempDist]
    C = np.argmin(EuclidianDistance, axis=1) + 1
    # step 2.b
    Y = {}
    for k in range(K):
        Y[k + 1] = np.array([]).reshape(19, 0)
    for i in range(m):
        Y[C[i]] = np.c_[Y[C[i]], X[i]]

    for k in range(K):
        Y[k + 1] = Y[k + 1].T

    for k in range(K):
        Centroids[:, k] = np.mean(Y[k + 1], axis=0)
Output = Y
cluster_centroids = Centroids.T

# concate the files
df = pd.concat([pd.DataFrame(train[i]), pd.DataFrame(labels)], axis=1)
df.columns = [*df.columns[:-1], 'label']
train_1 = df.sort_values(by='label')
# find covariance and weight
covr = []
weights = []
# ========================================================================================================#
# covariance all the train file
for name, group in train_1.groupby(by='label'):
    if (name == 0):
        covr.append(pd.DataFrame(np.cov(group.T)).iloc[:19, :-1])
        weights.append(len(group) / train_1.shape[0])
        gmm_covar.append(covr)
        each_weights.append(weights)
        cluster.append(centres)
return cluster, gmm_covar, each_weights, train_1, centres
# =========================================================================================================#
# probability each cluster
cluster, gmm_covar, each_weights, train_1, centres = calculate_mean_covariance(train)  # training gmm
gmm_covar = np.array(gmm_covar)  # covariance
each_weights = np.array(each_weights)  # probability weight function


# ==========================================================================================================#
# GMM
def GMM_tr(train):
    global cluster, gmm_covar, each_weights  # first three basics value called
    global last_mean, last_cov, last_weights  # second three basics value intialized
    last_mean = []
    last_cov = []
    last_weights = []
    for i in range(len(all_files)):
        gmm = GMM(5, 2)
        gmm1 = gmm.fit(train[i])
        last_mean.append(gmm1.mu)
        last_cov.append(gmm1.sigma)
        last_weights.append(gmm1.pi)
    return last_mean, last_cov, last_weights


last_mean, last_cov, last_weights = GMM_tr(train)
last_cov = np.array(last_cov)


# ===========================================================================================================#
# MAXIMIZATION STEP
def _m_step(X, gamma):
    N = X.shape[0]  # number of objects
    C = gamma.shape[1]  # number of clusters
    d = X.shape[1]  # dimension of each object
    # responsibilities for each gaussian
    sigma = []
    pi = np.mean(gamma, axis=0)
    mu = np.dot(gamma.T, X) / np.sum(gamma, axis=0)[:, np.newaxis]
    for c in range(C):
        x = X - mu[c, :]  # (N x d)
        gamma_diag = np.diag(gamma[:, c])
        x_mu = np.matrix(x)
        gamma_diag = np.matrix(gamma_diag)
        sigma_c = x.T * gamma_diag * x
        sigma.append(sigma_c)

        # print(log_likelihood)
    return pi, mu, sigma


# ==================================================================================================================#
# doing both expectation maximization alogorithm
def EM(train, cluster, gmm_covar, each_weights):
    arr = []
    for index, i in enumerate(train):
        arr1 = []
        for j, k, l in zip(i, cluster, gmm_covar):
            arr1.append(scipy.stats.multivariate_normal(mean=k, cov=l, allow_singular=True).pdf(j))
        arr.append(arr1)
    arr = np.array(arr)
    # weighted pdf(probability density function)
    a = np.array(each_weights) * arr
    # likelihood(taking maximum likehood function)
    likelihood_EM = np.sum(np.log(np.sum(a, axis=1)), axis=0)
    # gamma values
    gamma = a / np.sum(a, axis=1)[:, np.newaxis]
    print(gamma.shape)
    pi, mu, sigma = _m_step(train, gamma)
    for i in range(len(gamma)):
        likelihood_1 = []
        likelihood = math.log(np.sum(gamma))
        likelihood_1.append(likelihood)
        log_likelihood = np.sum(likelihood_1)
        print(log_likelihood)
        # maximum likehood first mean covariance weight
        gamma_last = ([[0 for j in range(centres.shape[0])] for i in range(train_1.shape[0])])
        gamma1_last1 = ([[0 for j in range(centres.shape[0])] for i in range(train_1.shape[0])])

        likelihood_second = []
        for i in range(len(gamma_last)):
            likelihood_last = math.log1p(np.sum(gamma_last[i]))
            likelihood_second.append(likelihood_last)
            log_likelihood_last = np.sum(likelihood_second)
            if log_likelihood == log_likelihood_last:
                print('Parameter updation completed')
                break
            else:
                print('Iteration needs to be done')
                return np.array(pi), np.array(mu), np.array(sigma), np.array(log_likelihood)


# ====================================================================================================================#
# last three basics function
last_weights = []  # final probiblity weight function
last_means = []  # final means
last_cov = []  # final covariance
for i in range(199):
    a, b, c, d = EM(train[i], cluster[i], gmm_covar[i], each_weights[i])  # accces em function
    last_weights.append(a)
    last_means.append(b)
    last_cov.append(c)
# ======================================================================================================================#
# find accuary
# arr1 has probabilities of each mfcc with all clusters
# find accuary
# arr1 has probabilities of each mfcc with all clusters
test = np.array(test)
arr1_test = []
gamma_test = ([[0 for j in range(centres.shape[0])] for i in range(test.shape[0])])
for index, i in enumerate(test):
    arr_test = []
    for j, k, l in zip(i, cluster[0], gmm_covar[0]):
        arr_test.append(scipy.stats.multivariate_normal(mean=k, cov=l, allow_singular=True).pdf(j))
    arr1_test.append(arr_test)
# for i in range(len(arr1_test)):
#     for j in range(len(arr1_test[i])):
#         gamma_test=each_weights[i][j] * arr1_test[i][j]
for i in range(len(arr1_test)):
    for j in range(len(arr1_test[0])):
        gamma_test[i][j] = each_weights[0][j] * arr1_test[i][j]
likelihood1_test = []
for i in range(len(gamma_test)):
    likelihood_test = math.log(np.sum(gamma_test))
    likelihood1_test.append(likelihood_test)
    log_likelihood_test = np.sum(likelihood1_test)
    print(log_likelihood_test)
# ========================================================================================#
final_prob = []
for i in range(200):
    prob_1 = np.argmax(log_likelihood_test[i])
    final_prob.append(prob_1)
count = 0
actual = list(range(200))
for i in range(200):
    if actual[i] == final_prob[i]:
        count += 1
        Accuracy = (count / 200) * 100
# ROC PLOT=========================================================================
from sklearn.metrics import confusion_matrix

Confusion_matrix = confusion_matrix(actual, final_prob)
print(confusion_matrix)

import sklearn.metrics as metrics

# calculate the fpr and tpr for all thresholds of the classification
probs = model.predict_proba(final_prob)
preds = probs[:, 1]
fpr, tpr, threshold = metrics.roc_curve(actual, preds)
roc_auc = metrics.auc(fpr, tpr)

# method I: plt
import matplotlib.pyplot as plt

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
plt.legend(loc='lower right')
plt.plot([0, 1], [0, 1], 'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()