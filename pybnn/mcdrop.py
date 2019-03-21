import time
import logging
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim

from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization
from pybnn.bayesian_linear_regression import BayesianLinearRegression, Prior


class Net(nn.Module):
    def __init__(self, n_inputs, dropout, n_units=[50, 50, 50]):
        super(Net, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.fc1 = nn.Linear(n_inputs, n_units[0])
        self.fc2 = nn.Linear(n_units[0], n_units[1])
        self.fc3 = nn.Linear(n_units[1], n_units[2])
        self.out = nn.Linear(n_units[2], 1)

    def forward(self, x):

        x = torch.tanh(self.fc1(x))
        x = self.dropout(x)

        x = torch.tanh(self.fc2(x))
        x = self.dropout(x)

        x = torch.tanh(self.fc3(x))
        x = self.dropout(x)

        return self.out(x)


class MCDROP(BaseModel):

    def __init__(self, batch_size=10, num_epochs=500,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 dropout = 0.05, tau = 1.0, T = 100,
                 normalize_input=True, normalize_output=True, rng=None):
        """
        This module performs MC Dropout for a fully connected
        feed forward neural network.

        Parameters
        ----------
        batch_size: int
            Batch size for training the neural network
        num_epochs: int
            Number of epochs for training
        learning_rate: float
            Initial learning rate for Adam
        adapt_epoch: int
            Defines after how many epochs the learning rate will be decayed by a factor 10
        n_units_1: int
            Number of units in layer 1
        n_units_2: int
            Number of units in layer 2
        n_units_3: int
            Number of units in layer 3
        dropoyt: float
            Dropout rate for all the dropout layers in the network.
        tau: float
            Tau value used for regularization
        T : int
            Number of samples for forward passes
        normalize_output : bool
            Zero mean unit variance normalization of the output values
        normalize_input : bool
            Zero mean unit variance normalization of the input values
        rng: np.random.RandomState
            Random number generator
        """

        if rng is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
        else:
            self.rng = rng

        self.X = None
        self.y = None
        self.network = None
        self.tau = tau
        self.dropout = dropout
        self.T = T
        self.normalize_input = normalize_input
        self.normalize_output = normalize_output

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.init_learning_rate = learning_rate

        self.n_units_1 = n_units_1
        self.n_units_2 = n_units_2
        self.n_units_3 = n_units_3
        self.adapt_epoch = adapt_epoch # TODO check
        self.network = None
        self.models = []

    @BaseModel._check_shapes_train
    def train(self, X, y):
        """
        Trains the model on the provided data.

        Parameters
        ----------
        X: np.ndarray (N, D)
            Input data points. The dimensionality of X is (N, D),
            with N as the number of points and D is the number of features.
        y: np.ndarray (N,)
            The corresponding target values.

        """
        start_time = time.time()

        # Normalize inputs
        if self.normalize_input:
            self.X, self.X_mean, self.X_std = zero_mean_unit_var_normalization(X)
        else:
            self.X = X

        # Normalize ouputs
        if self.normalize_output:
            self.y, self.y_mean, self.y_std = zero_mean_unit_var_normalization(y)
        else:
            self.y = y

        self.y = self.y[:, None]

        # Check if we have enough points to create a minibatch otherwise use all data points
        if self.X.shape[0] <= self.batch_size:
            batch_size = self.X.shape[0]
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]

        network = Net(n_inputs=features, dropout=self.dropout, n_units=[self.n_units_1, self.n_units_2, self.n_units_3])

        optimizer = optim.Adam(network.parameters(),
                               lr=self.init_learning_rate)

        # Start training
        lc = np.zeros([self.num_epochs])
        for epoch in range(self.num_epochs):

            epoch_start_time = time.time()

            train_err = 0
            train_batches = 0

            for batch in self.iterate_minibatches(self.X, self.y,
                                                  batch_size, shuffle=True):
                inputs = torch.Tensor(batch[0])
                targets = torch.Tensor(batch[1])

                optimizer.zero_grad()
                output = network(inputs)
                loss = torch.nn.functional.mse_loss(output, targets)
                loss.backward()
                optimizer.step()

                train_err += loss
                train_batches += 1

            lc[epoch] = train_err / train_batches
            logging.debug("Epoch {} of {}".format(epoch + 1, self.num_epochs))
            curtime = time.time()
            epoch_time = curtime - epoch_start_time
            total_time = curtime - start_time
            logging.debug("Epoch time {:.3f}s, total time {:.3f}s".format(epoch_time, total_time))
            logging.debug("Training loss:\t\t{:.5g}".format(train_err / train_batches))

        self.model = network

    def iterate_minibatches(self, inputs, targets, batchsize, shuffle=False):
        assert inputs.shape[0] == targets.shape[0], \
            "The number of training points is not the same"
        if shuffle:
            indices = np.arange(inputs.shape[0])
            self.rng.shuffle(indices)
        for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
            if shuffle:
                excerpt = indices[start_idx:start_idx + batchsize]
            else:
                excerpt = slice(start_idx, start_idx + batchsize)
            yield inputs[excerpt], targets[excerpt]

    @BaseModel._check_shapes_predict
    def predict(self, X_test):
        r"""
        Returns the predictive mean and variance of the objective function at
        the given test points.

        Parameters
        ----------
        X_test: np.ndarray (N, D)
            N input test points

        Returns
        ----------
        np.array(N,)
            predictive mean
        np.array(N,)
            predictive variance

        """
        # Normalize inputs
        if self.normalize_input:
            X_, _, _ = zero_mean_unit_var_normalization(X_test, self.X_mean, self.X_std)
        else:
            X_ = X_test

        # Perform MC dropout
        model = self.model
        T     = self.T

        # Yt_hat: T x N x 1
        Yt_hat = np.array([model(torch.Tensor(X_)).data.numpy() for _ in range(T)])
        # Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train  # T x N TODO check with Adam

        MC_pred_mean = np.mean(Yt_hat, 0)  # N x 1

        Second_moment = np.mean(Yt_hat ** 2, 0) # N x 1
        MC_pred_var = Second_moment + np.eye(Yt_hat.shape[-1]) / self.tau - (MC_pred_mean ** 2)

        m = MC_pred_mean.flatten()

        if MC_pred_var.shape[0] == 1:
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
        else:
            v = np.clip(MC_pred_var, np.finfo(MC_pred_var.dtype).eps, np.inf)
            v[np.where((v < np.finfo(v.dtype).eps) & (v > -np.finfo(v.dtype).eps))] = 0

        if self.normalize_output:
            m = zero_mean_unit_var_denormalization(m, self.y_mean, self.y_std)
            v *= self.y_std ** 2

        m = m.flatten()
        v = v.flatten()

        return m, v

    def get_incumbent(self):
        """
        Returns the best observed point and its function value

        Returns
        ----------
        incumbent: ndarray (D,)
            current incumbent
        incumbent_value: ndarray (N,)
            the observed value of the incumbent
        """

        inc, inc_value = super(MCDROP, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
