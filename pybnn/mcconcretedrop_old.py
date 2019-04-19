import time
import logging

import torch
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.autograd import Variable
import numpy as np
from pybnn.base_model import BaseModel
from pybnn.util.normalization import zero_mean_unit_var_normalization, zero_mean_unit_var_denormalization

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)


class ConcreteDropout(nn.Module):
    def __init__(self, weight_regularizer=1e-6,
                 dropout_regularizer=1e-5, init_min=0.1, init_max=0.1):
        super(ConcreteDropout, self).__init__()

        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer

        init_min = np.log(init_min) - np.log(1. - init_min)
        init_max = np.log(init_max) - np.log(1. - init_max)

        self.p_logit = nn.Parameter(torch.empty(1).uniform_(init_min, init_max))

    def forward(self, x, layer):
        p = torch.sigmoid(self.p_logit)

        out = layer(self._concrete_dropout(x, p))

        sum_of_square = 0
        for param in layer.parameters():
            sum_of_square += torch.sum(torch.pow(param, 2))

        weights_regularizer = self.weight_regularizer * sum_of_square / (1 - p)

        dropout_regularizer = p * torch.log(p)
        dropout_regularizer += (1. - p) * torch.log(1. - p)

        input_dimensionality = x[0].numel()  # Number of elements of first item in batch
        dropout_regularizer *= self.dropout_regularizer * input_dimensionality

        regularization = weights_regularizer + dropout_regularizer
        return out, regularization

    def _concrete_dropout(self, x, p):
        eps = 1e-7
        temp = 0.1

        unif_noise = torch.rand_like(x)

        drop_prob = (torch.log(p + eps)
                     - torch.log(1 - p + eps)
                     + torch.log(unif_noise + eps)
                     - torch.log(1 - unif_noise + eps))

        drop_prob = torch.sigmoid(drop_prob / temp)
        random_tensor = 1 - drop_prob
        retain_prob = 1 - p

        x = torch.mul(x, random_tensor)
        x /= retain_prob

        return x

class Net(nn.Module):
    def __init__(self, n_inputs, n_units=[50, 50, 50],
                 weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(n_inputs, n_units[0])
        self.linear2 = nn.Linear(n_units[0], n_units[1])
        self.linear3 = nn.Linear(n_units[1], n_units[2])
        self.out_mu = nn.Linear(n_units[2], 1)
        self.out_logvar = nn.Linear(n_units[2], 1)
        self.conc_drop1 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop2 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop3 = ConcreteDropout(weight_regularizer=weight_regularizer,
                                          dropout_regularizer=dropout_regularizer)
        self.conc_drop_mu = ConcreteDropout(weight_regularizer=weight_regularizer,
                                            dropout_regularizer=dropout_regularizer)
        self.conc_drop_logvar = ConcreteDropout(weight_regularizer=weight_regularizer,
                                                dropout_regularizer=dropout_regularizer)

        self.tanh = nn.Tanh()


    def forward(self, x):

        regularization = torch.empty(4, device=x.device)
        x1 = self.tanh(self.linear1(x))
        # x1, regularization[0] = self.conc_drop1(x, nn.Sequential(self.linear1, self.tanh))
        x2, regularization[0] = self.conc_drop2(x1, nn.Sequential(self.linear2, self.tanh))
        x3, regularization[1] = self.conc_drop3(x2, nn.Sequential(self.linear3, self.tanh))

        mean, regularization[2] = self.conc_drop_mu(x3, self.out_mu)
        log_var, regularization[3] = self.conc_drop_logvar(x3, self.out_logvar)

        return mean, log_var, regularization.sum()

class MCCONCRETEDROP(BaseModel):

    def __init__(self, batch_size=10, num_epochs=1000,
                 learning_rate=0.01,
                 adapt_epoch=5000, n_units_1=50, n_units_2=50, n_units_3=50,
                 length_scale = 1e-4, T = 100,
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
        self.length_scale = length_scale
        # self.weight_regularizer = weight_regularizer
        # self.dropout_regularizer = dropout_regularizer

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

        N = self.X.shape[0]


        # Check if we have enough points to create a minibatch otherwise use all data points
        if N <= self.batch_size:
            batch_size = N
        else:
            batch_size = self.batch_size

        # Create the neural network
        features = X.shape[1]
        wr = self.length_scale ** 2. / N
        dr = 2. / N
        network = Net(n_inputs=features, n_units=[self.n_units_1, self.n_units_2, self.n_units_3],
                      weight_regularizer=wr, dropout_regularizer=dr)

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

                # inputs = torch.Tensor(batch[0])
                # targets = torch.Tensor(batch[1])
                inputs  = Variable(torch.FloatTensor(batch[0]))
                targets = Variable(torch.FloatTensor(batch[1]))

                optimizer.zero_grad()
                output = network(inputs)
                loss = torch.nn.functional.mse_loss(output[0], targets)

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
        model.eval()
        # MC_samples : list T x N x 1
        # Yt_hat = np.array([model(torch.Tensor(X_)).data.numpy() for _ in range(T)])
        MC_samples = [model(Variable(torch.FloatTensor(X_))) for _ in range(T)]
        means = torch.stack([tup[0] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()
        logvar = torch.stack([tup[1] for tup in MC_samples]).view(T, X_.shape[0]).data.numpy()
        # Yt_hat = Yt_hat * self.std_y_train + self.mean_y_train  # T x N TODO check with Adam
        aleatoric_uncertainty = np.exp(logvar).mean(0)
        # tau =
        epistemic_uncertainty = np.var(means, 0).mean(0)

        MC_pred_mean = np.mean(means, 0)  # N x 1

        Second_moment = np.mean(means ** 2, 0) # N x 1
        MC_pred_var = Second_moment + epistemic_uncertainty - (MC_pred_mean ** 2)

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

        inc, inc_value = super(MCCONCRETEDROP, self).get_incumbent()
        if self.normalize_input:
            inc = zero_mean_unit_var_denormalization(inc, self.X_mean, self.X_std)

        if self.normalize_output:
            inc_value = zero_mean_unit_var_denormalization(inc_value, self.y_mean, self.y_std)

        return inc, inc_value
