import numpy as np
import pickle
import matplotlib.pyplot as plt


class LogisticRegression:
    def __init__(self, a=5, b=5):
        """
        y_hat = sigmoid(a + b * x)

        """
        self.a = a
        self.b = b

    def train_gradient_descent(self, X, Y, max_iter=1000, learning_rate=1):
        loss = []
        a_loss = []
        b_loss = []
        acc = []
        curr_iter = 0
        m = len(X)

        while curr_iter < max_iter:
            curr_iter += 1
            Y_hat = self.sigmoid(self.a + self.b * X)

            curr_loss = self.cross_entropy(X=X, Y=Y, a=self.a, b=self.b)
            loss.append(curr_loss)

            a_loss.append(self.a)
            b_loss.append(self.b)

            # update parameter
            gradient_a = 1/m * sum((Y_hat - Y))
            gradient_b = 1/m * sum((Y_hat - Y) * X)

            self.a -= learning_rate * gradient_a
            self.b -= learning_rate * gradient_b

            # get acc
            curr_acc = self.metrics(X=X, Y=Y, op='Accuracy')
            acc.append(curr_acc)

        return loss, a_loss, b_loss, acc

    def cross_entropy(self, X, Y, a, b):
        Y_hat = self.sigmoid(a + b * X)
        return -1/X.shape[0] * sum((Y * np.log(Y_hat)) + (1 - Y) * (np.log(1 - Y_hat)))

    def classifier(self, X):
        """
        if sigmoid(X * Theta) > 0.5:
            return 1
        else
            return 0
        """
        Y_hat_value = self.sigmoid(self.a + self.b * X)
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
        pickle.dump({'a': self.a, 'b': self.b}, open(path, 'wb'))

    def load(self, path):
        parameter = pickle.load(open(path, 'rb'))
        self.a = parameter['a']
        self.b = parameter['b']

    def visual_loss(self, x, y, a_loss, b_loss, curr_loss):
        fig = plt.figure()
        ax3 = plt.axes(projection='3d')

        a = np.arange(-10, 10, 0.5)
        b = np.arange(-10, 10, 0.5)
        la = len(a)
        lb = len(b)

        A, B = np.meshgrid(a, b)

        loss = np.zeros((lb, la), dtype=float)
        for i in range(lb):
            for j in range(la):
                loss[i][j] = self.cross_entropy(X=x, Y=y, a=A[i][j], b=B[i][j])

        ax3.scatter3D(a_loss, b_loss, curr_loss, color="red", s=20)
        ax3.plot_surface(A, B, loss, cmap='rainbow', alpha=0.6)

        plt.xlabel("a")
        plt.ylabel("b")
        plt.title("Loss Function")

        plt.show()

    @staticmethod
    def sigmoid(X):
        return 1 / (1 + np.exp(-X))

    @staticmethod
    def extend(X):
        return np.c_[np.ones(len(X)), X]


def main():
    x = np.array(
        [137.97, 104.50, 100.00, 126.32, 79.20, 99.00, 124.00, 114.00, 106.69, 140.05, 53.75, 46.91, 68.00, 63.02,
         81.26, 86.21])
    x = (x - np.min(x)) / (np.max(x) - np.min(x))
    y = np.array([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0])

    colors = ['b', 'r']

    lr = LogisticRegression()
    loss, a, b, acc = lr.train_gradient_descent(X=x, Y=y, max_iter=200, learning_rate=10)

    plt.figure(0)
    for i in range(x.shape[0]):
        plt.scatter(x[i], y[i], c=colors[y[i]])

    plt.title("PREDICT")
    plt.xlabel("x")
    plt.ylabel("y")
    testx = np.linspace(0, 1, 100)
    plt.plot(testx, lr.sigmoid(lr.a + lr.b * testx))


    plt.figure(1)
    plt.title("LOSS")
    plt.xlabel("iterate")
    plt.ylabel("loss")
    plt.plot(np.linspace(0, len(loss), len(loss)), loss, color='red')


    plt.figure(2)
    plt.title("ACC")
    plt.xlabel("iterate")
    plt.ylabel("acc")
    plt.plot(np.linspace(0, len(acc), len(acc)), acc, color='red')
    plt.ylim((0, 1))

    lr.visual_loss(x, y, a, b, loss)

    plt.show()


if __name__ == '__main__':
    main()







