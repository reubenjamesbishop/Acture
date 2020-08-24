
import pandas as pd
from utils import Data

df = pd.read_csv('../data/advertising.csv')
a = Data(df)

# Support Vector Machine investigation
from sklearn import svm
clf = svm.SVC()
clf.fit(a.X_train, a.y_train)
predictions_SVM = clf.predict(a.X_test)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(a.y_test,predictions_SVM))
print(confusion_matrix(a.y_test,predictions_SVM))
