import numpy as np
from sklearn.cluster import KMeans

class DeepRVFL:
    """A deep RVFL classifier or regression."""

    def __init__(self, n_nodes, k, lam, w_random_vec_range, b_random_vec_range, activation, n_layer, same_feature=False,
                 task_type='classification'):
        assert task_type in ['classification', 'regression'], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.k = k  # Número de clusters
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = []
        self.random_bias = []
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.n_layer = n_layer
        self.data_std = [None] * self.n_layer
        self.data_mean = [None] * self.n_layer
        self.same_feature = same_feature
        self.task_type = task_type

    def train(self, data, label, n_class):
        n_sample = len(data)
        n_feature = len(data[0])
        d = self.standardize(data, 0)  # Normalization data
        h = data.copy()
        
        # Aplicar KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        kmeans.fit(h)
        cluster_labels = kmeans.labels_

        for i in range(self.n_layer):
            h = self.standardize(h, i)  # Normalization data
            self.random_weights.append(self.get_random_vectors(len(h[0]), self.n_nodes, self.w_random_range))
            self.random_bias.append(self.get_random_vectors(1, self.n_nodes, self.b_random_range))
            
            # Lista para almacenar los h de cada cluster
            h_list = []
            
            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self.activation_function(np.dot(cluster_data, self.random_weights[i]) + 
                                                     np.dot(np.ones([n_cluster_sample, 1]), self.random_bias[i]))
                h_list.append(h_cluster)
            
            # Concatenar todos los h para obtener una matriz h única
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, d], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        if self.task_type == 'classification':
            y = self.one_hot(label, n_class)
        else:
            y = label
        if n_sample > (self.n_nodes * self.n_layer + n_feature):
            self.beta = np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d))).dot(d.T).dot(y)
        else:
            self.beta = d.T.dot(np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))).dot(y)

    def predict(self, data):
        n_sample = len(data)
        d = self.standardize(data, 0)
        h = data.copy()

        # Aplicar KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        kmeans.fit(h)
        cluster_labels = kmeans.labels_

        for i in range(self.n_layer):
            h = self.standardize(h, i)
            
            # Lista para almacenar los h de cada cluster
            h_list = []
            
            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self.activation_function(np.dot(cluster_data, self.random_weights[i]) + 
                                                     np.dot(np.ones([n_cluster_sample, 1]), self.random_bias[i]))
                h_list.append(h_cluster)
            
            # Concatenar todos los h para obtener una matriz h única
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, d], axis=1)
            
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == 'regression':
            return output

    def eval(self, data, label):
        n_sample = len(data)
        d = self.standardize(data, 0)
        h = data.copy()

        # Aplicar KMeans
        kmeans = KMeans(n_clusters=self.k, n_init=10)
        kmeans.fit(h)
        cluster_labels = kmeans.labels_

        for i in range(self.n_layer):
            h = self.standardize(h, i)
            
            # Lista para almacenar los h de cada cluster
            h_list = []
            
            for j in range(self.k):
                cluster_data = h[cluster_labels == j]
                n_cluster_sample = len(cluster_data)
                h_cluster = self.activation_function(np.dot(cluster_data, self.random_weights[i]) + 
                                                     np.dot(np.ones([n_cluster_sample, 1]), self.random_bias[i]))
                h_list.append(h_cluster)
            
            # Concatenar todos los h para obtener una matriz h única
            final_h = np.concatenate(h_list, axis=0)
            h = final_h
            d = np.concatenate([h, d], axis=1)
            
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == 'classification':
            result = np.argmax(output, axis=1)
            acc = np.sum(np.equal(result, label)) / len(label)
            return acc
        elif self.task_type == 'regression':
            mae = np.mean(np.abs(output - label))
            return mae

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[0]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x, index):
        if self.same_feature is True:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x)
            return (x - self.data_mean[index]) / self.data_std[index]
        else:
            if self.data_std[index] is None:
                self.data_std[index] = np.maximum(np.std(x, axis=0), 1/np.sqrt(len(x)))
            if self.data_mean[index] is None:
                self.data_mean[index] = np.mean(x, axis=0)
            return (x - self.data_mean[index]) / self.data_std[index]

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat((np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1)


class Activation:
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.e ** (-x))

    @staticmethod
    def sine(x):
        return np.sin(x)

    @staticmethod
    def hardlim(x):
        return (np.sign(x) + 1) / 2

    @staticmethod
    def tribas(x):
        return np.maximum(1 - np.abs(x), 0)

    @staticmethod
    def radbas(x):
        return np.exp(-(x**2))

    @staticmethod
    def sign(x):
        return np.sign(x)

    @staticmethod
    def relu(x):
        return np.maximum(0, x)


