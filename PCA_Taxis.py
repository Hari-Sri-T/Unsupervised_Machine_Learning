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
#Taking the Taxi's Dataset
taxis_df = sns.load_dataset('taxis')
taxis_df.head()


#%%
#Data Presprocessing
#%%
taxis_df.describe()

# %%
taxis_df.shape

# %%
taxis_df.isnull().sum()

#%%

df = taxis_df.dropna()  # Drop rows with missing values

#%%


# Converting Target Variable to Numerial
le = LabelEncoder()
taxis_df['payment'] = le.fit_transform(taxis_df['payment'])


# %%


# Select features for classification
X = taxis_df[['fare', 'distance', 'pickup_borough', 'dropoff_borough']]
X = pd.get_dummies(X, drop_first=True)  # Encode categorical features
y = taxis_df['payment']  # Target (Cash or Card)


# %%
# Train-Test Split


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
# %%
# Scaling the features


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %%


# Training a Classifier on Raw Data
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_train)
# %%

y_pred = clf.predict(X_test_scaled)
print("Accuracy on Raw Data:", metrics.accuracy_score(y_test, y_pred))


# %%
##Applying PCA (2 Components)
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# %%
## Creating ClaSsfier using pca2
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)

# %%
y_pred_pca2 = clf_pca.predict(X_test_pca)
print("Accuracy with PCA 2 components:", metrics.accuracy_score(y_test, y_pred_pca2))
# %%
## Applying PCA ( 3 Components)
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)
# %%
clf_pca = RandomForestClassifier(n_estimators=100, random_state=42)
clf_pca.fit(X_train_pca, y_train)

# %%
y_pred_pca3 = clf_pca.predict(X_test_pca)
print("Accuracy with PCA 3 components:", metrics.accuracy_score(y_test, y_pred_pca3))

# %%
print("\nThe Accuracy of all is Respectively:\n")

print("Accuracy on Raw Data:", metrics.accuracy_score(y_test, y_pred))
print("Accuracy with PCA 2 components:", metrics.accuracy_score(y_test, y_pred_pca2))
print("Accuracy with PCA 3 components:", metrics.accuracy_score(y_test, y_pred_pca3))
