#!/usr/bin/env python

# Deep Learning Homework 1

import math
import argparse
import numpy as np
import matplotlib.pyplot as plt
import utils
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

relu = lambda z : np.maximum(0,z)
sign = lambda x: 1 if x >= 0 else -1


class LinearModel(object):
    def __init__(self, n_classes, n_features, **kwargs):
        self.W = np.zeros((n_classes, n_features))

    def update_weight(self, x_i, y_i, **kwargs):
        raise NotImplementedError

    def train_epoch(self, X, y, **kwargs):
        for x_i, y_i in zip(X, y):
            self.update_weight(x_i, y_i, **kwargs)

    def predict(self, X):
        """X (n_examples x n_features)"""
        scores = np.dot(self.W, X.T)  # (n_classes x n_examples)
        predicted_labels = scores.argmax(axis=0)  # (n_examples)
        return predicted_labels

    def evaluate(self, X, y):
        """
        X (n_examples x n_features):
        y (n_examples): gold labels
        """
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible


class Perceptron(LinearModel):

    def update_weight(self, x_i, y_i, **kwargs):
        """
        x_i (n_features): a single training example
        y_i (scalar): the gold label for that example
        other arguments are ignored
        """
        # Q1.1a
        eta = 1
        _y = np.argmax(np.dot(self.W, x_i))
        if _y != y_i:
            self.W[y_i, :] += eta*x_i
            self.W[_y, :] -= eta*x_i


class LogisticRegression(LinearModel):

    softmax = lambda z : np.exp(z) / np.sum(np.exp(z))
    def update_weight(self, x_i, y_i, learning_rate=0.001):
        """
        x_i (n_features): a single training example
        y_i: the gold label for that example
        learning_rate (float): keep it at the default value for your plots
        """
        # Q1.1b
        y_onehot = np.array([1 if y_i==i else 0 for i in range(4)])
        z = LogisticRegression.softmax(np.dot(self.W, x_i))
        self.W += learning_rate * np.dot(np.expand_dims(y_onehot - z, axis=1), np.expand_dims(x_i, axis=0))


class MLP(object):
    # Q3.2b. This MLP skeleton code allows the MLP to be used in place of the
    # linear models with no changes to the training loop or evaluation code
    # in main().

    def softmax(z):
        z2 = z - np.max(z, axis=1, keepdims=True)
        return np.exp(z2) / np.sum(np.exp(z2), axis=1, keepdims=True)

    def __init__(self, n_classes, n_features, hidden_size):
        # Initialize an MLP with a single hidden layer.
        self.n_features = n_features
        self.hidden_size = hidden_size
        self.W = [np.random.normal(0.1,0.1,size=(hidden_size, n_features)), 
                  np.random.normal(0.1,0.1,size=(n_classes, hidden_size))]
        self.b = [np.zeros(hidden_size), np.zeros(n_classes)]

    def predict(self, X):
        # Compute the forward pass of the network. At prediction time, there is
        # no need to save the values of hidden nodes, whereas this is required
        # at training time.
        h = relu(np.dot(X,self.W[0].T) + self.b[0])
        o = MLP.softmax(np.dot(h,self.W[1].T) + self.b[1])
        return np.argmax(o, axis=1)
        


    def evaluate(self, X, y):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        """
        # Identical to LinearModel.evaluate()
        y_hat = self.predict(X)
        n_correct = (y == y_hat).sum()
        n_possible = y.shape[0]
        return n_correct / n_possible

    def train_epoch(self, X, y, learning_rate=0.001):
        """
        Dont forget to return the loss of the epoch.
        """
        d_relu = lambda z : np.where(z > 0, 1, 0)
        loss = 0
        for i in range(0, X.shape[0]):
            z1 = np.dot(X[i,:],self.W[0].T) + self.b[0]
            a1 = relu(z1)
            z2 = np.dot(a1,self.W[1].T) + self.b[1]
            a2 = softmax(z2[None,:])

            y_ = np.array([1 if j == y[i] else 0 for j in range(4)])
            d_z2 = a2 - y_
            d_W2 = np.dot(a1[:,None], d_z2).T
            d_b2 = np.sum(d_z2, axis=0, keepdims=True)
            d_z1 = np.dot(d_z2, self.W[1]) * d_relu(z1)
            d_W1 = np.dot(X[i,:][:,None], d_z1).T
            d_b1 = np.sum(d_z1, axis=0, keepdims=True)

            self.W[0] -= learning_rate * d_W1
            self.W[1] -= learning_rate * d_W2
            self.b[0] -= learning_rate * d_b1.reshape(-1)
            self.b[1] -= learning_rate * d_b2.reshape(-1)

            loss += -np.dot(y_[None,:],np.log(a2).T)
        loss = loss[0,0]

        return loss



def plot(epochs, train_accs, val_accs):
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.plot(epochs, train_accs, label='train')
    plt.plot(epochs, val_accs, label='validation')
    plt.legend()
    plt.show()

def plot_loss(epochs, loss):
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.plot(epochs, loss, label='train')
    plt.legend()
    plt.show()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model',
                        choices=['perceptron', 'logistic_regression', 'mlp'],
                        help="Which model should the script run?")
    parser.add_argument('-epochs', default=20, type=int,
                        help="""Number of epochs to train for. You should not
                        need to change this value for your plots.""")
    parser.add_argument('-hidden_size', type=int, default=200,
                        help="""Number of units in hidden layers (needed only
                        for MLP, not perceptron or logistic regression)""")
    parser.add_argument('-learning_rate', type=float, default=0.001,
                        help="""Learning rate for parameter updates (needed for
                        logistic regression and MLP, but not perceptron)""")
    opt = parser.parse_args()

    utils.configure_seed(seed=42)

    add_bias = opt.model != "mlp"
    data = utils.load_oct_data(bias=add_bias)
    train_X, train_y = data["train"]
    dev_X, dev_y = data["dev"]
    test_X, test_y = data["test"]
    n_classes = np.unique(train_y).size
    n_feats = train_X.shape[1]

    # initialize the model
    if opt.model == 'perceptron':
        model = Perceptron(n_classes, n_feats)
    elif opt.model == 'logistic_regression':
        model = LogisticRegression(n_classes, n_feats)
    else:
        model = MLP(n_classes, n_feats, opt.hidden_size)
    epochs = np.arange(1, opt.epochs + 1)
    train_loss = []
    valid_accs = []
    train_accs = []
    
    for i in epochs:
        print('Training epoch {}'.format(i))
        train_order = np.random.permutation(train_X.shape[0])
        train_X = train_X[train_order]
        train_y = train_y[train_order]
        if opt.model == 'mlp':
            loss = model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        else:
            model.train_epoch(
                train_X,
                train_y,
                learning_rate=opt.learning_rate
            )
        
        train_accs.append(model.evaluate(train_X, train_y))
        valid_accs.append(model.evaluate(dev_X, dev_y))
        if opt.model == 'mlp':
            print('loss: {:.4f} | train acc: {:.4f} | val acc: {:.4f}'.format(
                loss, train_accs[-1], valid_accs[-1],
            ))
            train_loss.append(loss)
        else:
            print('train acc: {:.4f} | val acc: {:.4f}'.format(
                 train_accs[-1], valid_accs[-1],
            ))
    print('Final test acc: {:.4f}'.format(
        model.evaluate(test_X, test_y)
        ))

    # plot
    plot(epochs, train_accs, valid_accs)
    if opt.model == 'mlp':
        plot_loss(epochs, train_loss)


if __name__ == '__main__':
    main()
