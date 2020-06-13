import numpy as np
import pandas as pd
from io import StringIO
from scipy.special import expit as sigmoid
from scipy.special import softmax

IRIS_DATASET = """sepal_length,sepal_width,petal_length,petal_width,species
                5.1,3.5,1.4,0.2,Iris-setosa
                4.9,3.0,1.4,0.2,Iris-setosa
                4.7,3.2,1.3,0.2,Iris-setosa
                4.6,3.1,1.5,0.2,Iris-setosa
                5.0,3.6,1.4,0.2,Iris-setosa
                5.4,3.9,1.7,0.4,Iris-setosa
                4.6,3.4,1.4,0.3,Iris-setosa
                5.0,3.4,1.5,0.2,Iris-setosa
                4.4,2.9,1.4,0.2,Iris-setosa
                4.9,3.1,1.5,0.1,Iris-setosa
                5.4,3.7,1.5,0.2,Iris-setosa
                4.8,3.4,1.6,0.2,Iris-setosa
                4.8,3.0,1.4,0.1,Iris-setosa
                4.3,3.0,1.1,0.1,Iris-setosa
                5.8,4.0,1.2,0.2,Iris-setosa
                5.7,4.4,1.5,0.4,Iris-setosa
                5.4,3.9,1.3,0.4,Iris-setosa
                5.1,3.5,1.4,0.3,Iris-setosa
                5.7,3.8,1.7,0.3,Iris-setosa
                5.1,3.8,1.5,0.3,Iris-setosa
                5.4,3.4,1.7,0.2,Iris-setosa
                5.1,3.7,1.5,0.4,Iris-setosa
                4.6,3.6,1.0,0.2,Iris-setosa
                5.1,3.3,1.7,0.5,Iris-setosa
                4.8,3.4,1.9,0.2,Iris-setosa
                5.0,3.0,1.6,0.2,Iris-setosa
                5.0,3.4,1.6,0.4,Iris-setosa
                5.2,3.5,1.5,0.2,Iris-setosa
                5.2,3.4,1.4,0.2,Iris-setosa
                4.7,3.2,1.6,0.2,Iris-setosa
                4.8,3.1,1.6,0.2,Iris-setosa
                5.4,3.4,1.5,0.4,Iris-setosa
                5.2,4.1,1.5,0.1,Iris-setosa
                5.5,4.2,1.4,0.2,Iris-setosa
                4.9,3.1,1.5,0.1,Iris-setosa
                5.0,3.2,1.2,0.2,Iris-setosa
                5.5,3.5,1.3,0.2,Iris-setosa
                4.9,3.1,1.5,0.1,Iris-setosa
                4.4,3.0,1.3,0.2,Iris-setosa
                5.1,3.4,1.5,0.2,Iris-setosa
                5.0,3.5,1.3,0.3,Iris-setosa
                4.5,2.3,1.3,0.3,Iris-setosa
                4.4,3.2,1.3,0.2,Iris-setosa
                5.0,3.5,1.6,0.6,Iris-setosa
                5.1,3.8,1.9,0.4,Iris-setosa
                4.8,3.0,1.4,0.3,Iris-setosa
                5.1,3.8,1.6,0.2,Iris-setosa
                4.6,3.2,1.4,0.2,Iris-setosa
                5.3,3.7,1.5,0.2,Iris-setosa
                5.0,3.3,1.4,0.2,Iris-setosa
                7.0,3.2,4.7,1.4,Iris-versicolor
                6.4,3.2,4.5,1.5,Iris-versicolor
                6.9,3.1,4.9,1.5,Iris-versicolor
                5.5,2.3,4.0,1.3,Iris-versicolor
                6.5,2.8,4.6,1.5,Iris-versicolor
                5.7,2.8,4.5,1.3,Iris-versicolor
                6.3,3.3,4.7,1.6,Iris-versicolor
                4.9,2.4,3.3,1.0,Iris-versicolor
                6.6,2.9,4.6,1.3,Iris-versicolor
                5.2,2.7,3.9,1.4,Iris-versicolor
                5.0,2.0,3.5,1.0,Iris-versicolor
                5.9,3.0,4.2,1.5,Iris-versicolor
                6.0,2.2,4.0,1.0,Iris-versicolor
                6.1,2.9,4.7,1.4,Iris-versicolor
                5.6,2.9,3.6,1.3,Iris-versicolor
                6.7,3.1,4.4,1.4,Iris-versicolor
                5.6,3.0,4.5,1.5,Iris-versicolor
                5.8,2.7,4.1,1.0,Iris-versicolor
                6.2,2.2,4.5,1.5,Iris-versicolor
                5.6,2.5,3.9,1.1,Iris-versicolor
                5.9,3.2,4.8,1.8,Iris-versicolor
                6.1,2.8,4.0,1.3,Iris-versicolor
                6.3,2.5,4.9,1.5,Iris-versicolor
                6.1,2.8,4.7,1.2,Iris-versicolor
                6.4,2.9,4.3,1.3,Iris-versicolor
                6.6,3.0,4.4,1.4,Iris-versicolor
                6.8,2.8,4.8,1.4,Iris-versicolor
                6.7,3.0,5.0,1.7,Iris-versicolor
                6.0,2.9,4.5,1.5,Iris-versicolor
                5.7,2.6,3.5,1.0,Iris-versicolor
                5.5,2.4,3.8,1.1,Iris-versicolor
                5.5,2.4,3.7,1.0,Iris-versicolor
                5.8,2.7,3.9,1.2,Iris-versicolor
                6.0,2.7,5.1,1.6,Iris-versicolor
                5.4,3.0,4.5,1.5,Iris-versicolor
                6.0,3.4,4.5,1.6,Iris-versicolor
                6.7,3.1,4.7,1.5,Iris-versicolor
                6.3,2.3,4.4,1.3,Iris-versicolor
                5.6,3.0,4.1,1.3,Iris-versicolor
                5.5,2.5,4.0,1.3,Iris-versicolor
                5.5,2.6,4.4,1.2,Iris-versicolor
                6.1,3.0,4.6,1.4,Iris-versicolor
                5.8,2.6,4.0,1.2,Iris-versicolor
                5.0,2.3,3.3,1.0,Iris-versicolor
                5.6,2.7,4.2,1.3,Iris-versicolor
                5.7,3.0,4.2,1.2,Iris-versicolor
                5.7,2.9,4.2,1.3,Iris-versicolor
                6.2,2.9,4.3,1.3,Iris-versicolor
                5.1,2.5,3.0,1.1,Iris-versicolor
                5.7,2.8,4.1,1.3,Iris-versicolor
                6.3,3.3,6.0,2.5,Iris-virginica
                5.8,2.7,5.1,1.9,Iris-virginica
                7.1,3.0,5.9,2.1,Iris-virginica
                6.3,2.9,5.6,1.8,Iris-virginica
                6.5,3.0,5.8,2.2,Iris-virginica
                7.6,3.0,6.6,2.1,Iris-virginica
                4.9,2.5,4.5,1.7,Iris-virginica
                7.3,2.9,6.3,1.8,Iris-virginica
                6.7,2.5,5.8,1.8,Iris-virginica
                7.2,3.6,6.1,2.5,Iris-virginica
                6.5,3.2,5.1,2.0,Iris-virginica
                6.4,2.7,5.3,1.9,Iris-virginica
                6.8,3.0,5.5,2.1,Iris-virginica
                5.7,2.5,5.0,2.0,Iris-virginica
                5.8,2.8,5.1,2.4,Iris-virginica
                6.4,3.2,5.3,2.3,Iris-virginica
                6.5,3.0,5.5,1.8,Iris-virginica
                7.7,3.8,6.7,2.2,Iris-virginica
                7.7,2.6,6.9,2.3,Iris-virginica
                6.0,2.2,5.0,1.5,Iris-virginica
                6.9,3.2,5.7,2.3,Iris-virginica
                5.6,2.8,4.9,2.0,Iris-virginica
                7.7,2.8,6.7,2.0,Iris-virginica
                6.3,2.7,4.9,1.8,Iris-virginica
                6.7,3.3,5.7,2.1,Iris-virginica
                7.2,3.2,6.0,1.8,Iris-virginica
                6.2,2.8,4.8,1.8,Iris-virginica
                6.1,3.0,4.9,1.8,Iris-virginica
                6.4,2.8,5.6,2.1,Iris-virginica
                7.2,3.0,5.8,1.6,Iris-virginica
                7.4,2.8,6.1,1.9,Iris-virginica
                7.9,3.8,6.4,2.0,Iris-virginica
                6.4,2.8,5.6,2.2,Iris-virginica
                6.3,2.8,5.1,1.5,Iris-virginica
                6.1,2.6,5.6,1.4,Iris-virginica
                7.7,3.0,6.1,2.3,Iris-virginica
                6.3,3.4,5.6,2.4,Iris-virginica
                6.4,3.1,5.5,1.8,Iris-virginica
                6.0,3.0,4.8,1.8,Iris-virginica
                6.9,3.1,5.4,2.1,Iris-virginica
                6.7,3.1,5.6,2.4,Iris-virginica
                6.9,3.1,5.1,2.3,Iris-virginica
                5.8,2.7,5.1,1.9,Iris-virginica
                6.8,3.2,5.9,2.3,Iris-virginica
                6.7,3.3,5.7,2.5,Iris-virginica
                6.7,3.0,5.2,2.3,Iris-virginica
                6.3,2.5,5.0,1.9,Iris-virginica
                6.5,3.0,5.2,2.0,Iris-virginica
                6.2,3.4,5.4,2.3,Iris-virginica
                5.9,3.0,5.1,1.8,Iris-virginica"""

class iris_dataset():
    def __init__(self):
        self.df = pd.read_csv(StringIO(IRIS_DATASET))
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
