# Please make sure you convert the original spambase.data file into .csv format. Check out convert-file-type in my repo for instructions.
# or you could download the CSV file I've added.

# Identifying if an email is spam or not and visualizing their volumes

# Importing libraries

import pandas as pd, numpy as np

# Importing dataset

dataset = pd.read_csv('spambase.csv')
dataset = dataset.sample(frac = 1)  # Random shuffling of the observations

# Dividing dataset into independent feature matrix and target variable vector

X = dataset.iloc[:, : -1].values
y = dataset.iloc[:,  -1].values

# Training & testing datasets

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2,
                                                    random_state = 1)

# Classification Model

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 12, random_state = 1)
classifier.fit(X_train, y_train)

# Prediction

y_pred = classifier.predict(X_test)

# Evaluation metrics to judge model performance

from sklearn.metrics import accuracy_score, recall_score, r2_score, precision_score
accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# Applying k-Fold Cross Validation to check for model performance 

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
accuracies.std()
accuracies.mean()

# Visualizing spam and non-spam email volumes
import seaborn as sb

k = []
for i in y_pred:
    if i == 1:
        k.append(['Spam Email'])
    else:
        k.append(['Non-Spam Email'])

k = pd.DataFrame(k)
sb.countplot(x = k.iloc[:, 0].values, data = k, palette = 'PuBuGn')
