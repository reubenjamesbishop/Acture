import numpy as np
import sklearn
import pandas as pd

# Data

df = pd.read_csv('../data/advertising.csv')
X = df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
y = df['Clicked on Ad']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# Support Vector Machine
from sklearn import svm
clf = svm.SVC()
clf.fit(X_train,y_train)
predictions_SVM = clf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,predictions_SVM))
print(confusion_matrix(y_test,predictions_SVM))
