import numpy as np
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

class Data:

    def __init__(self, dataset):

        self.df = dataset
        self.X, self.y = self.define_features()
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def define_features(self):
        """Function to split data based on current harcoded features."""

        X = self.df[['Daily Time Spent on Site','Age','Area Income','Daily Internet Usage','Male']]
        y = self.df['Clicked on Ad']

        return X, y

    def split_data(self):
        """Split training and testing data with sklearn tool"""

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=101)

        return X_train, X_test, y_train, y_test
