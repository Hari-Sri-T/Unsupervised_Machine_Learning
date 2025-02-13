#%%
# Importing Required Libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn import metrics
#%%
#Taking the Iris's Dataset
iris_df = sns.load_dataset('iris')
iris_df.head()


#%%
#Data Presprocessing
#%%
iris_df.describe()

# %%
iris_df.shape

# %%
iris_df.isnull().sum()

#%%

df = iris_df.dropna()  # Drop rows with missing values
 #This won't do anything as there is no null values in the dataset
#%%

#Selecting the features
X = iris_df.drop(columns=['species'])  # Features
y = iris_df['species']  # Target



# %%

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# %%

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# %%

# Train classifier on raw data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
y_pred = clf.predict(X_test_scaled)
print("Accuracy on Raw Data:", metrics.accuracy_score(y_test, y_pred))

# %%

# Apply PCA (2 Components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%

# Train classifier with PCA-transformed data
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca2 = clf_pca.predict(X_test_pca)
print("Accuracy with PCA (2 components):", metrics.accuracy_score(y_test, y_pred_pca2))

# %%

# Apply PCA (3 Components)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


# %%
# Train classifier with PCA-transformed data
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca3 = clf_pca.predict(X_test_pca)
print("Accuracy with PCA (3 components):", metrics.accuracy_score(y_test, y_pred_pca3))

# %%

print (" Accuracy's of all: \n")

print("Accuracy on Raw Data:", metrics.accuracy_score(y_test, y_pred))

print("Accuracy with PCA (2 components):", metrics.accuracy_score(y_test, y_pred_pca2))

print("Accuracy with PCA (3 components):", metrics.accuracy_score(y_test, y_pred_pca3))

