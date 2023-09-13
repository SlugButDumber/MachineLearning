#Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2

# Importing the dataset
from sklearn.datasets import load_digits
dataset = load_digits()
X = dataset.data
y = dataset.target
print(X.shape)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Training Models

# Training the Logistic Regression model on the Training set
from sklearn.linear_model import LogisticRegression
classifier_1 = LogisticRegression()
classifier_1.fit(X_train, y_train)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier_2 = GaussianNB()
classifier_2.fit(X_train, y_train)

# Training the SVM model on the Training set
from sklearn.svm import SVC
classifier_3 = SVC(kernel = 'linear')
classifier_3.fit(X_train, y_train)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier_4 = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier_4.fit(X_train, y_train)

# Training the Kernel SVM model on the Training set
from sklearn.svm import SVC
classifier_5 = SVC(kernel = 'rbf')
classifier_5.fit(X_train, y_train)

# Training the Decision Tree Classification model on the Training set
from sklearn.tree import DecisionTreeClassifier
classifier_6 = DecisionTreeClassifier(criterion = 'entropy')
classifier_6.fit(X_train, y_train)

# Training the Random Forest Classification model on the Training set
from sklearn.ensemble import RandomForestClassifier
classifier_7 = RandomForestClassifier(n_estimators = 10, criterion = 'entropy')
classifier_7.fit(X_train, y_train)

#Accuracy scores of models

# Logistic Regression Accuracy
from sklearn.metrics import accuracy_score
y_pred = classifier_1.predict(X_test)
print('Logistic Regression Accuracy: ', accuracy_score(y_test, y_pred))

# Naive Bayes Accuracy
y_pred = classifier_2.predict(X_test)
print('Naive Bayes Accuracy: ', accuracy_score(y_test, y_pred))

# SVM Accuracy
y_pred = classifier_3.predict(X_test)
print('SVM Accuracy: ', accuracy_score(y_test, y_pred))

# K-NN Accuracy
y_pred = classifier_4.predict(X_test)
print('K-NN Accuracy: ', accuracy_score(y_test, y_pred))

# Kernel SVM Accuracy
y_pred = classifier_5.predict(X_test)
print('Kernel SVM Accuracy: ', accuracy_score(y_test, y_pred))

# Decision Tree Accuracy
y_pred = classifier_6.predict(X_test)
print('Decision Tree Accuracy: ', accuracy_score(y_test, y_pred))

# Random Forest Accuracy
y_pred = classifier_7.predict(X_test)
print('Random Forest Accuracy: ', accuracy_score(y_test, y_pred))

# plt.imshow(sc.inverse_transform(X_test)[0].reshape(8,8), cmap='gray')
# plt.show()

#Testing real data
img1 = cv2.imread("C:\Drive D\ArhatPersonal\ML\Practice\Proj_MNIST\Testing_3.png",0)
plt.imshow(img1, cmap='gray')
plt.show()

img = cv2.resize(img1,(8,8))
plt.imshow(img, cmap='gray')
plt.show()

print(classifier_5.predict(sc.transform(img.reshape(1,64))))