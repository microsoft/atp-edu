import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle


class LinearRegression:

    def __init__(self, a=0, b=0):
        """
        y_hat = a + b * x

        """
        self.a = a
        self.b = b

    def train_gradient_descent(self, x, y, max_iter=100, min_loss=0.01, learning_rate=0.001):
        """
        梯度下降法

        error = 1 / 2 * (a + b * x - y) ^ 2

        loss = 1 / (2m) * sum((a + b * x - y) ^ 2)

        gradient_a = 1 / (m) * sum(a + b * x - y)

        gradient_b = 1 / (m) * sum((a + b * x - y) * x)
        """
        loss = []
        a_loss = []
        b_loss = []
        curr_loss = float('inf')
        curr_iter = 0
        m = len(x)

        while curr_iter < max_iter and curr_loss > min_loss:
            curr_iter += 1
            curr_loss = 1/(2 * m) * sum((self.a + self.b * x - y) ** 2)

            a_loss.append(self.a)
            b_loss.append(self.b)
            loss.append(curr_loss)

            gradient_a = 1/(2 * m) * sum(self.a + self.b * x - y)
            gradient_b = 1/(2 * m) * sum((self.a + self.b * x - y) * x)

            self.a -= learning_rate * gradient_a
            self.b -= learning_rate * gradient_b

        return loss, a_loss, b_loss

    def train_least_square_method(self, x, y):
        """
        最小二乘法

        b = (sum(x*y) - 1/m * sum(x) * sum(y)) /(sum(x^2) - 1/m * sum(x)*sum(y))

        a = 1/m * sum(y) - b/m * sum(x)

        """
        m = len(x)

        self.b = (sum(x * y) - 1/m * sum(x) * sum(y)) / (sum(x * x) - 1/m * sum(x) * sum(x))
        self.a = 1/m * sum(y) - self.b/m * sum(x)

    def gen_visual_data(self, arr, element_num, scale):
        max = np.max(arr)
        min = np.min(arr)

        offset = (max - min) * scale

        max += offset
        min -= offset

        interval = (max - min) / (element_num * 1.0)

        return np.arange(min, max, interval)


    def visual_loss(self, x, y, a_loss, b_loss, curr_loss):
        fig = plt.figure()
        ax3 = plt.axes(projection='3d')

        #a = np.arange(-0.05, 0.05, 0.005)
        #b = np.arange(-0.03, 0.2, 0.005)

        a = self.gen_visual_data(a_loss, 10, 0.2)
        b = self.gen_visual_data(b_loss, 40, 2.0)

        print(a)
        print(b)

        la = len(a)
        lb = len(b)

        A, B = np.meshgrid(a, b)

        loss = np.zeros((lb, la), dtype=float)
        for i in range(lb):
            for j in range(la):
                loss[i][j] = 1/(2 * len(x)) * sum((A[i][j] + B[i][j] * x - y) ** 2)

        ax3.scatter3D(a_loss, b_loss, curr_loss, color="red", s=20)
        ax3.plot_surface(A, B, loss, cmap='rainbow', alpha=0.6)

        plt.xlabel("a")
        plt.ylabel("b")
        plt.title("Loss Function")
        plt.show()

    def predict(self, x):
        return self.a + self.b * x

    def metrics(self, op, x, y):
        m = len(x)
        y_hat = self.predict(x)

        print(str(op)+": ")
        if op == 'MAE':
            return 1/m * sum(np.abs(y - y_hat))
        elif op == 'MSE':
            return 1/m * sum((y - y_hat) ** 2)
        elif op == 'RMSE':
            return np.sqrt(1/m * sum((y - y_hat) ** 2))
        elif op == 'MAPE':
            return np.median(np.abs(y - y_hat)/y)
        elif op == 'R2':
            return 1 - sum((y_hat - y) ** 2) / sum((y - np.mean(y_hat)) ** 2)

    def save(self, path):
        pickle.dump({'a': self.a, 'b': self.b}, open(path, 'wb'))

    def load(self, path):
        parameter = pickle.load(open(path, 'rb'))
        self.a = parameter['a']
        self.b = parameter['b']
        return

    def train_with_visualization(self, x, y):
        loss, a, b = self.train_gradient_descent(x=x, y=y, max_iter=100, learning_rate=0.01)

        print(self.metrics('MAE', x, y))
        plt.figure(1)
        plt.plot(np.linspace(0, len(loss), len(loss)), loss, color='red')
        plt.xlabel("iteration")
        plt.ylabel("loss")

        plt.figure(2)
        plt.scatter(x, y)
        plt.plot(x, self.predict(x), color='red')
        plt.xlabel("x")
        plt.ylabel("y")

        return a, b, loss


def train_with_visualization(x, y):
    lr = LinearRegression()
    loss, a, b = lr.train_gradient_descent(x=x, y=y, max_iter=100, learning_rate=0.01)

    print(lr.metrics('MAE', x, y))
    plt.figure(1)
    plt.plot(np.linspace(0, len(loss), len(loss)), loss, color='red')
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.figure(2)
    plt.scatter(x, y)
    plt.plot(x, lr.predict(x), color='red')
    plt.xlabel("x")
    plt.ylabel("y")

    lr.visual_loss(x, y, a, b, loss)

    return


if __name__ == '__main__':
    data = pd.read_csv('./winequality-red.csv', sep=';')
    col = data.columns

    x = data[col[0]].to_numpy()
    y = data[col[-3]].to_numpy()
    train_with_visualization(x, y)
