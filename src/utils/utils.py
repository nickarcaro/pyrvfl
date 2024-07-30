import numpy as np


### random vector generation ###
def get_random_vectors(m, n, scale_range):
    x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
    return x


### one hot encoding to classification ###
def one_hot(x, n_class):
    y = np.zeros([len(x), n_class])
    for i in range(len(x)):
        y[i, x[i]] = 1
    return y


### standarize data rvfl ###
def standardize_rvfl(self, x):
    if self.same_feature is True:
        if self.data_std is None:
            self.data_std = np.maximum(np.std(x), 1 / np.sqrt(len(x)))
        if self.data_mean is None:
            self.data_mean = np.mean(x)
        return (x - self.data_mean) / self.data_std
    else:
        if self.data_std is None:
            self.data_std = np.maximum(np.std(x, axis=0), 1 / np.sqrt(len(x)))
        if self.data_mean is None:
            self.data_mean = np.mean(x, axis=0)
        return (x - self.data_mean) / self.data_std


### standardize data to Deep Models ###
def standardize(self, x, index):
    if self.same_feature is True:
        if self.data_std[index] is None:
            self.data_std[index] = np.maximum(np.std(x), 1 / np.sqrt(len(x)))
        if self.data_mean[index] is None:
            self.data_mean[index] = np.mean(x)
        return (x - self.data_mean[index]) / self.data_std[index]
    else:
        if self.data_std[index] is None:
            self.data_std[index] = np.maximum(np.std(x, axis=0), 1 / np.sqrt(len(x)))
        if self.data_mean[index] is None:
            self.data_mean[index] = np.mean(x, axis=0)
        return (x - self.data_mean[index]) / self.data_std[index]
