import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


class LinearRegression:

    def __init__(self, Theta=[], n=1):
        """
        theta: 参数
        n: feature数

        y_hat = theta0 + theta1 * x1 + theta2 * x2 + ··· + thetan * xn

        令 x0 = 1

        y_hat = theta0 * x0 + theta1 * x1 + ···+ thetan * xn

        Y = X * Theta
        """
        if Theta == []:
            self.n = n
            self.Theta = np.array([np.ones(n + 1)]).T
        else:
            self.Theta = Theta
            self.n = len(Theta) + 1

    def train_gradient_descent_loop(self, X, Y, max_iter=20, min_loss=0.1, learning_rate=0.001):
        """
        循环更新参数

        y_hat = theta0 * x0 + theta1 * x1 + ··· + thetan * xn

        error = 1/2 *(y_hat - y) ^ 2

        J(theta) = 1/(2m) * sum_i(y_hat[i] - y[i]) ^2

        gradient theta[k] = 1/(m) * sum_i(y_hat[i] - y[i]) * theta[k]

        """
        loss = []
        curr_iter = 0
        curr_loss = float('inf')
        m = len(X)
        X_train = self.extend(X)

        while curr_iter < max_iter and curr_loss > min_loss:
            curr_iter += 1
            gradient = np.zeros(self.n + 1)
            curr_loss = 0

            # loss
            for i in range(m):
                curr_loss += 1/(2 * m) * ((np.dot(X_train[i], self.Theta) - Y[i][0]) ** 2)

            loss.append(curr_loss)

            # update parameter
             # calculate gradient
            for k in range(self.n+1):
                for i in range(m):
                    gradient[k] += (X_train[i][k]/m) * (np.dot(X_train[i], self.Theta) - Y[i][0])
             # update parameter
            for k in range(self.n+1):
                self.Theta[k] -= learning_rate * gradient[k]

        return loss

    def train_gradient_descent_matrix(self, X, Y, max_iter=20, min_loss=0.1, learning_rate=0.0001):
        """
        矩阵乘法更新参数

        Y_hat = X * Theta

        J(Theta) = 1/(2*m) * (X*Theta - Y).T * (X*Theta - Y)

        """
        loss = []
        curr_iter = 0
        curr_loss = float('inf')
        m = len(X)
        X_train = self.extend(X)

        while curr_iter < max_iter and curr_loss > min_loss:
            curr_iter += 1

            # loss
            residual = self.predict(X) - Y
            curr_loss = 1/(2 * m) * np.matmul(residual.T, residual)[0][0]
            loss.append(curr_loss)

            # update parameter
            gradient = 1/m * np.matmul(X_train.T, residual)
            self.Theta -= learning_rate * gradient

        return loss

    def train_least_square_method(self, X, Y):
        """
        解析法求解
        """
        X_train = self.extend(X)
        self.Theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_train.T, X_train)), X_train.T), Y)

    def predict(self, x):
        return np.matmul(self.extend(x), self.Theta)

    def metrics(self, op, x, y):
        m = len(x)
        y_hat = self.predict(x)

        print(str(op) + ": ")
        if op == 'MAE':
            return 1 / m * sum(np.abs(y - y_hat))
        elif op == 'MSE':
            return 1 / m * sum((y - y_hat) ** 2)
        elif op == 'RMSE':
            return np.sqrt(1 / m * sum((y - y_hat) ** 2))
        elif op == 'MAPE':
            return np.median(np.abs(y - y_hat) / y)
        elif op == 'R2':
            return 1 - sum((y_hat - y) ** 2) / sum((y - np.mean(y_hat)) ** 2)

    def save(self, path):
        pickle.dump(self.Theta, open(path, 'wb'))

    def load(self, path):
        self.Theta = pickle.load(open(path, 'rb'))
        self.n = len(self.Theta) - 1

    @staticmethod
    def extend(X):
        return np.c_[np.ones(len(X)), X]

def main():
    data = pd.read_csv('./winequality-red.csv', sep=';')
    data_np = data.to_numpy()

    X = data_np[:, :-1]
    Y = np.array([data_np[:, -1]]).T

    lr = LinearRegression(n=X.shape[1])

    lr.train_least_square_method(X, Y)

    print("=========LSM:=========")
    print(lr.metrics(op='MSE', x=X, y=Y))

    learning_rate = [0.0001,0.0002, 0.0005]
    for l_r in learning_rate:
        #plt.figure(1)
        #lr = LinearRegression(n=X.shape[1])
        #loss = lr.train_gradient_descent_loop(X, Y, learning_rate=l_r)
        #plt.plot(np.linspace(0, len(loss), len(loss)), loss,)
        
        plt.figure(2)
        lr1 = LinearRegression(n=X.shape[1])
        loss = lr1.train_gradient_descent_matrix(X, Y, max_iter=500, learning_rate=l_r)

        loss = loss[100:]
        plt.plot(np.linspace(100, len(loss), len(loss)), loss)
        print("=========GD:learning rate= {} =========".format(l_r))
        print(lr1.metrics(op='MSE', x=X, y=Y))

    plt.legend(["learning rate ={}".format(str(l_r)) for l_r in learning_rate])
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.show()


if __name__ == '__main__':
    main()