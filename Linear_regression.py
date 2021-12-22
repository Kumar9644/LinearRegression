import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class LinearRegression:
    def __init__(self, lr=0.01, n_iters=1000):
        self.lr = lr
        self.n_iters = n_iters
        self.weights = None
        self.bias = None

    def fit(self, x, y):
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        for _ in range(self.n_iters):
            y_predicted = np.dot(x, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(x.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, x):
        y_predicted = np.dot(x, self.weights) + self.bias
        return y_predicted

if __name__=='__main__':
    #mean squared error
    def mse(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)

    #r2_score
    def r2_score(y_test,y_pred):
        corr=np.corrcoef(y_test,y_pred)
        corr=corr[0,1]
        return corr**2
    X,y=datasets.make_regression(n_samples=100,n_features=1,noise=10,random_state=4)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    reg = LinearRegression(lr=0.01)
    reg.fit(x_train, y_train)
    y_pred = reg.predict(x_test)
    print(r2_score(y_test,y_pred))
