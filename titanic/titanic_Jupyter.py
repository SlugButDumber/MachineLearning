# %% [markdown]
# # Importing Libraries

# %%
#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% [markdown]
# # Training Set

# %% [markdown]
# # Reading Dataset

# %%
dataset = pd.read_csv("C:/Drive D/ArhatPersonal/ML/Kaggle/titanic/train.csv")
X_train = dataset.iloc[:, [2,4,5,6,7,9]].values
y_train = dataset.iloc[:, 1].values

# %%
print(X_train)

# %% [markdown]
# # Taking care of missing data

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_train[:, [2]])
X_train[:, [2]] = imputer.transform(X_train[:, [2]])

# %%
print(X_train)

# %% [markdown]
# # Encoding categorical data

# %%
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
X_train[:, 1] = le.fit_transform(X_train[:, 1])


# %%
print(X_train)

# %% [markdown]
# # Feature Scaling

# %%
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, [2,3,4,5]] = sc.fit_transform(X_train[:, [2,3,4,5]])

# %%
print(X_train)

# %% [markdown]
# # Training the model

# %%
from sklearn.svm import SVC
classifier = SVC(kernel='rbf', random_state=0)
classifier.fit(X_train, y_train)

# %% [markdown]
# # Confusion Matrix and Accuracy

# %%
from sklearn.metrics import confusion_matrix
y_pred_train = classifier.predict(X_train)
cm = confusion_matrix(y_train, y_pred_train)
print(cm)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_train, y_pred_train))

# %% [markdown]
# # Test Set

# %% [markdown]
# # Importing Test Set

# %%
Test_dataset = pd.read_csv("C:/Drive D/ArhatPersonal/ML/Kaggle/titanic/test.csv")
X_test = Test_dataset.iloc[:, [1,3,4,5,6,8]].values

# %%
print(X_test)

# %% [markdown]
# # Taking Care of Missing Data

# %%
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X_test[:, [2]])
X_test[:, [2]] = imputer.transform(X_test[:, [2]])

# %%
print(X_test)

# %% [markdown]
# # Encoding categorical data

# %%
X_test[:, 1] = le.transform(X_test[:, 1])

# %%
print(X_test)

# %% [markdown]
# # Feature Scaling

# %%
X_test[:, [2,3,4,5]] = sc.transform(X_test[:, [2,3,4,5]])

# %%
print(X_test)

# %% [markdown]
# # Predicting Test Set results

# %%
y_pred = classifier.predict(X_test)

# %% [markdown]
# # Creating Submission File

# %%
submission = pd.DataFrame({'PassengerId':Test_dataset['PassengerId'],'Survived':y_pred})
submission.to_csv('submission.csv',index=False)


