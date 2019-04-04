# Perceptron practice

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import perceptron
from sklearn.linear_model import Perceptron

def main():
    # loading data
    iris = load_iris()
    df = pd.DataFrame(iris.data)
    df['label'] = iris.target

    df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
    print(df.label.value_counts())

    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    data = np.array(df.iloc[:100, [0, 1, -1]])
    X, y = data[:, :-1], data[:, -1]
    y = np.array([1 if i == 1 else -1 for i in y])

    # method-1: Perceptron achieved by stochastic gradient descent
    perceptron_m = perceptron.Model(len(data[0]))
    perceptron_m.fit(X, y)

    x_points = np.linspace(4, 7, 2)
    y_ = -(perceptron_m.w[0] * x_points + perceptron_m.b) / perceptron_m.w[1]
    plt.plot(x_points, y_)
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()

    # method-2: Scikit-learn Perceptron
    clf = Perceptron(fit_intercept=False, max_iter=1000, shuffle=False)
    clf.fit(X, y)
    # Weights assigned to the featrues
    print(clf.coef_)
    # 截距 Constants in decision function
    print(clf.intercept_)
    x_points = np.arange(4, 8)
    y_ = -(clf.coef_[0][0] * x_points + clf.intercept_) / clf.coef_[0][1]
    plt.plot(x_points, y_)
    plt.scatter(df[:50]['sepal length'], df[:50]['sepal width'], label='0')
    plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
