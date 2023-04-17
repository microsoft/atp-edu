import random
import math
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, Theta=[], n=1):
        """
        y_hat = sigmoid(theta0 * x0 + theta1 * x1 + ··· + thetan * xn)

        """
        if Theta == []:
            self.n = n
            self.Theta = np.array([np.ones(n + 1)]).T
        else:
            self.Theta = Theta
            self.n = len(Theta) + 1
        pass

    def train_gradient_descent(self, X, Y, max_iter=500, learning_rate=0.5):
        X_train = self.extend(X)
        loss = []
        acc = []
        curr_iter = 0
        m = len(X)

        while curr_iter < max_iter:
            curr_iter += 1
            Y_hat = self.sigmoid(np.matmul(X_train, self.Theta))

            curr_loss = self.cross_entropy(X, Y)
            loss.append(curr_loss)

            # update parameter
            gradient = 1 / m * np.matmul(X_train.T, Y_hat - Y)
            self.Theta -= learning_rate * gradient

            # get acc
            curr_acc = self.metrics(X=X, Y=Y, op='Accuracy')
            acc.append(curr_acc)

        return loss, acc

    # 交叉熵损失函数
    def cross_entropy(self, X, Y):
        Y_hat = self.sigmoid(np.matmul(self.extend(X), self.Theta))
        return -1/X.shape[0] * (np.matmul(Y.T, np.log(Y_hat)) + np.matmul((1-Y).T, np.log(1 - Y_hat)))[0][0]

    def train_mini_batch_gradient_descent(self, X, Y, batch_size=32, epoch=10, learning_rate=0.001):
        X_train = self.extend(X)
        loss = []
        acc = []
        m = len(X)

        iterations = math.ceil(m/batch_size)
        for ep in range(epoch):
            for iteration in range(iterations):
                curr_loss = self.cross_entropy(X, Y)
                loss.append(curr_loss)

                if iteration < iterations:
                    x = X_train[iteration * batch_size: (iteration+1) * batch_size]
                    y = Y[iteration * batch_size: (iteration+1) * batch_size]
                else:
                    x = X_train[iteration * batch_size:]
                    y = Y[iteration * batch_size:]

                Y_hat = self.sigmoid(np.matmul(x, self.Theta))

                gradient = 1 / x.shape[0] * np.matmul(x.T, Y_hat - y)
                self.Theta -= learning_rate * gradient

                # get acc
                curr_acc = self.metrics(X=X, Y=Y, op='Accuracy')
                acc.append(curr_acc)

        return loss, acc

    def train_stochastic_gradient_descent(self, X, Y, max_iter=500, learning_rate=0.5):
        X_train = self.extend(X)
        loss = []
        acc = []
        curr_iter = 0
        m = len(X)

        while curr_iter < max_iter:
            curr_iter += 1
            Y_hat = self.sigmoid(np.matmul(X_train, self.Theta))

            curr_loss = self.cross_entropy(X, Y)
            loss.append(curr_loss)

            # update parameter
             # 随机数选取随机样本
            random.seed(random.random())
            sample = random.randint(0, m-1)
             # 随机样本产生的梯度
            gradient = 1 / m * np.array([X_train[sample]]).T * (Y_hat[sample] - Y[sample])
            self.Theta -= learning_rate * gradient

            # get acc
            curr_acc = self.metrics(X=X, Y=Y, op='Accuracy')
            acc.append(curr_acc)

        return loss, acc

    def classifier(self, X):
        """
        if sigmoid(X * Theta) > 0.5:
            return 1
        else
            return 0
        """

        Y_hat_value = self.sigmoid(np.matmul(self.extend(X), self.Theta))
        return np.array([[1] if y_hat > 0.5 else [0] for y_hat in Y_hat_value])

    def metrics(self, op, X, Y):
        """
        Precision = TP / (TP + FP)
        Recall = TP / (TP + FN)
        Accuracy = (TP + TN) / (TP + TN + FP + FN)
        F1 = 1 / (1 / Precision + 1 / Recall)
        """
        Y_hat = self.classifier(X)
        TP, FP, TN, FN = 0, 0, 0, 0
        for i in range(len(Y_hat)):
            if Y[i] == 1:
                if Y_hat[i] == 1:
                    TP += 1
                else:
                    FN += 1
            if Y[i] == 0:
                if Y_hat[i] == 0:
                    TN += 1
                else:
                    FP += 1

        if op == 'F1':
            1 / ((TP + FP)/TP + (TP + FN)/TP)
        elif op == 'Precision':
            TP / (TP + FP)
        elif op == 'Recall':
            return TP / (TP + FN)
        elif op == 'Accuracy':
            return (TP + TN) / (TP + TN + FP + FN)

    def save(self, path):
        pickle.dump(self.Theta, open(path, 'wb'))

    def load(self, path):
        self.Theta = pickle.load(open(path, 'rb'))
        self.n = len(self.Theta) - 1

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def extend(X):
        return np.c_[np.ones(len(X)), X]


# normalization
def normalize(X):
    for i in range(X.shape[1]):
        X[:, i] = (X[:, i] - np.min(X[:, i])) / (np.max(X[:, i]) - np.min(X[:, i]))


def main():
    data = pd.read_csv('./heart.csv')
    data_np = data.to_numpy()

    random.seed(0.5)
    random.shuffle(data_np)

    X = np.array(data_np[:, :-1])
    Y = np.array([data_np[:, -1]]).T

    normalize(X)
    lr = LogisticRegression(n=X.shape[1])

    #loss, acc = lr.train_gradient_descent(X, Y, learning_rate=10, max_iter=100)
    loss, acc = lr.train_stochastic_gradient_descent(X, Y, learning_rate=10, max_iter=2000)
    #loss, acc = lr.train_mini_batch_gradient_descent(X, Y, learning_rate=10, batch_size=303, epoch=100)

    plt.figure(1)
    plt.plot(np.linspace(0, len(loss), len(loss)), loss, color='red')
    plt.title("LOSS")
    plt.xlabel("iteration")
    plt.ylabel("loss")

    plt.figure(2)
    plt.plot(np.linspace(0, len(acc), len(acc)), acc, color='red')
    plt.ylim((0, 1))
    plt.title("ACC")
    plt.xlabel("iteration")
    plt.ylabel("acc")
    plt.show()


if __name__ == '__main__':
    main()







