"Import Libraries"

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"Load Dataset"

heart_data = pd.read_csv('heart.csv')
heart_data.head()

"Check Dataset Information"

heart_data.shape
heart_data.info()
heart_data.describe()

"Check Target Values"

heart_data['target'].value_counts()

"Split Features and Target"

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

"Split Training and Testing Data"

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

"Train The Model"

print(heart_data['target'].value_counts())

from sklearn.model_selection import train_test_split

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=2)

print(heart_data['target'].value_counts())

heart_data.loc[0:5,'target'] = 0
heart_data.loc[6:,'target'] = 1

X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)

"Train model"

from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)

"Training Accuracy"

from sklearn.metrics import accuracy_score

X_train_prediction = model.predict(X_train)
training_accuracy = accuracy_score(Y_train, X_train_prediction)

print("Training Accuracy:", training_accuracy)

"Testing Accuracy"

X_test_prediction = model.predict(X_test)
test_accuracy = accuracy_score(Y_test, X_test_prediction)

print("Testing Accuracy:", test_accuracy)

"Heart Disease Prediction System"

import numpy as np

input_data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)

input_data_as_numpy_array = np.asarray(input_data)

input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)

if prediction[0] == 0:
    print("The person does not have Heart Disease")
else:
    print("The person has Heart Disease")
