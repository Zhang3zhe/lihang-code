# KNN practice

import knn
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def L(x, y, p=2):
    if len(x) == len(y) and len(x) > 1:
        sum = 0
        for i in range(len(x)):
            sum += math.pow(abs(x[i]-y[i]), p)
        return math.pow(sum, 1/p)
    else:
        return 0


def main():
    # Loading data
    iris = load_iris()
    df = pd.DataFrame(iris.data)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = knn.KNN(X_train, y_train)
    clf.score(X_test, y_test)

    test_point = [6.0, 3.0]
    print('Test Point: {}'.format(clf.predict(test_point)))

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.plot(test_point[0], test_point[1], 'bo', label='test_point')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    # Methond-2 sklearn
    clf_sk = KNeighborsClassifier()
    clf_sk.fit(X_train, y_train)
    clf_sk.score(X_test, y_test)

if __name__ == '__main__':
    main()
