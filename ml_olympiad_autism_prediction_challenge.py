# -*- coding: utf-8 -*-
"""ML Olympiad - Autism Prediction Challenge.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1051sOx8y2cCPbILkwsK_SyAxgiVeWDt-
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler

import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('/content/train.csv')
print(df.head())

df.shape

df.info()

df.describe().T

df['ethnicity'].value_counts()

df['relation'].value_counts()

df = df.replace({'yes':1, 'no':0, '?':'Others', 'others':'Others'})

plt.pie(df['Class/ASD'].value_counts().values, autopct='%1.1f%%')
plt.show()

ints = []
objects = []
floats = []

for col in df.columns:
  if df[col].dtype == int:
    ints.append(col)
  elif df[col].dtype == object:
    objects.append(col)
  else:
    floats.append(col)

ints.remove('ID')
ints.remove('Class/ASD')

plt.subplots(figsize=(15,15))

for i, col in enumerate(ints):
  plt.subplot(4,3,i+1)
  sb.countplot(df[col], hue=df['Class/ASD'])
plt.tight_layout()
plt.show()

import seaborn as sb
import matplotlib.pyplot as plt
import math

# Assuming you have a list of integer columns to plot (ints)
# For example:


# Assuming 'Class/ASD' is the target variable, and 'df' is your DataFrame

# Calculate the number of rows and columns based on the number of columns you want to plot
n_cols = 5  # Number of columns in the grid
n_rows = math.ceil(len(ints) / n_cols)  # Calculate the number of rows based on the number of columns

plt.figure(figsize=(15, 15))

# Create a grid of count plots for each integer column
for i, col in enumerate(ints):
    plt.subplot(n_rows, n_cols, i + 1)
    ax = sb.countplot(x=col, data=df, hue='Class/ASD', ax=plt.gca())
    ax.set_title(col)

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,5))
sb.countplot(data=df, x='country_of_res', hue='Class/ASD')
plt.xticks(rotation=90)
plt.show()

plt.subplots(figsize=(15,5))

for i, col in enumerate(floats):
  plt.subplot(1,2,i+1)
  sb.distplot(df[col])
  plt.tight_layout()
  plt.show()

plt.subplots(figsize=(15,5))

for i, col in enumerate(floats):
  plt.subplot(1,2,i+1)
  sb.boxplot(df[col])
plt.tight_layout()
plt.show()

df = df[df['result']>-5]
df.shape

# This functions make groups by taking
# the age as a parameter
def convertAge(age):
	if age < 4:
		return 'Toddler'
	elif age < 12:
		return 'Kid'
	elif age < 18:
		return 'Teenager'
	elif age < 40:
		return 'Young'
	else:
		return 'Senior'

df['ageGroup'] = df['age'].apply(convertAge)

sb.countplot(x=df['ageGroup'], hue=df['Class/ASD'])
plt.show()

def add_feature(data):

# Creating a column with all values zero
  data['sum_score'] = 0
  for col in data.loc[:,'A1_Score':'A10_Score'].columns:

    # Updating the 'sum_score' value with scores
    # from A1 to A10
    data['sum_score'] += data[col]

  # Creating a random data using the below three columns
  data['ind'] = data['austim'] + data['used_app_before'] + data['jaundice']

  return data

df = add_feature(df)

sb.countplot(x=df['sum_score'], hue=df['Class/ASD'])
plt.show()

# Applying log transformations to remove the skewness of the data.
df['age'] = df['age'].apply(lambda x: np.log(x))

sb.distplot(df['age'])
plt.show()

from sklearn.preprocessing import LabelEncoder
import seaborn as sb
import matplotlib.pyplot as plt

def encode_labels(data):
    for col in data.columns:
        if data[col].dtype == 'object':
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])
    return data

# Assuming you have a DataFrame 'df'
df = encode_labels(df)

# Create a correlation matrix
correlation_matrix = df.corr()

# Set a threshold for correlation values to highlight
threshold = 0.8  # You can adjust this threshold as needed

# Create a mask to hide the upper triangle of the correlation matrix
mask = correlation_matrix.where(abs(correlation_matrix) > threshold, 0)

# Making a heatmap to visualize the correlation matrix
plt.figure(figsize=(10, 10))
sb.heatmap(mask, annot=True, cbar=False, cmap='coolwarm')  # Adjust the colormap as needed
plt.show()

removal = ['ID', 'age_desc', 'used_app_before', 'austim']
features = df.drop(removal + ['Class/ASD'], axis=1)
target = df['Class/ASD']

X_train, X_val, Y_train, Y_val = train_test_split(features, target, test_size = 0.2, random_state=10)

# As the data was highly imbalanced we will balance it by adding repetitive rows of minority class.
ros = RandomOverSampler(sampling_strategy='minority',random_state=0)
X, Y = ros.fit_resample(X_train,Y_train)
X.shape, Y.shape

# Normalizing the features for stable and fast training.
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

models = [LogisticRegression(), XGBClassifier(), SVC(kernel='rbf')]

for model in models:
  model.fit(X, Y)

  print(f'{model} : ')
  print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
  print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))
  print()