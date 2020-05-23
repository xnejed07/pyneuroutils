import numpy as np
import pandas as pd
from scipy.special import expit as sigmoid
from scipy.special import softmax


class iris_dataset():
    def __init__(self):
        self.df = pd.read_csv('./miscellaneous/tests_iris_dataset.csv')
        targets = self.df['species'].unique().tolist()
        self.df['targets'] = self.df.apply(lambda x: targets.index(x['species']), axis=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        X = self.df.iloc[item][['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy(
            dtype='float').reshape(1, 4)
        y = self.df.iloc[item]['targets']
        return X, y

    def get_full_dataset(self):
        X = self.df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']].to_numpy(dtype='float')
        y = self.df['targets'].to_numpy()
        return X, y


def iris_classifier(X):
    theta = np.array([[0.48994069, 1.43882369, -2.16677303, -1.01059485, 0.27273983],
                      [0.58435622, -0.01273847, -0.54079646, -0.98260182, 0.40343509],
                      [-0.86247388, -1.0294514, 1.39310749, 1.13095965, -0.51583529]])
    data = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    probs = np.dot(theta, data.T)
    probs = softmax(probs.T, axis=1)
    return probs


class diabetes_dataset():
    def __init__(self):
        self.df = pd.read_csv('./miscellaneous/tests_pima_indians_diabetes.csv')

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        X = self.df.iloc[item][
            ['num_pregnant', 'plasma_glucose', 'diastolic_pressure', 'skin_thickness', 'insulin', 'BMI',
             'diabetes_pedigree_function', 'age']].to_numpy(dtype='float').reshape(1, 4)
        y = self.df.iloc[item]['targets']
        return X, y

    def get_full_dataset(self):
        X = self.df[['num_pregnant', 'plasma_glucose', 'diastolic_pressure', 'skin_thickness', 'insulin', 'BMI',
                     'diabetes_pedigree_function', 'age']].to_numpy(dtype='float')
        y = self.df['targets'].to_numpy()
        return X, y


def diabetes_classifier(X):
    theta = np.array([[1.17252327e-01, 3.35997542e-02, -1.40873997e-02, -1.27050855e-03, -1.24031661e-03,
                       7.72024358e-02, 1.41904141e+00, 1.00354444e-02, -7.70290616]])
    X = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    probs = np.dot(theta, X.T)
    probs = sigmoid(probs.T)
    probs = np.concatenate([1 - probs, probs], axis=1)
    return probs
