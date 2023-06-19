#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the dataset
dataset = pd.read_csv("C:/Drive D/ArhatPersonal/ML/Kaggle/titanic/train.csv")
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y_train = dataset.iloc[:, 1].values

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X[:, [2,3,4,5]] = sc.fit_transform(X[:, [2,3,4,5]])

#training the model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X, y_train)

#Training Set

#confusion matrix
from sklearn.metrics import confusion_matrix
y_pred = classifier.predict(X)
cm = confusion_matrix(y_train, y_pred)
print(cm)

#accuracy
from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_pred))


#Test set

#reading the test dataset
Test_dataset = pd.read_csv("C:/Drive D/ArhatPersonal/ML/Kaggle/titanic/test.csv")
X_test = Test_dataset.iloc[:, [1,3,4,5,6,8]].values

#taking care of missing data
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

#encoding categorical data
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X[:, 1] = le.fit_transform(X[:, 1])

#feature scaling
X_test[:, [2,3,4,5]] = sc.transform(X_test[:, [2,3,4,5]])

#predicting the test set results
y_pred_test = classifier.predict(X_test)

#creating the submission file
submission = pd.DataFrame({'PassengerId':Test_dataset['PassengerId'],'Survived':y_pred_test})
submission.to_csv('submission.csv',index=False)