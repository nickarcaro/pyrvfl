import numpy as np


class SPRVFL:
    """A simple RVFL classifier or regression.

    Attributes:
        n_nodes: An integer of enhancement node number.
        lam: A floating number of regularization parameter.
        w_random_vec_range: A list, [min, max], the range of generating random weights.
        b_random_vec_range: A list, [min, max], the range of generating random bias.
        random_weights: A Numpy array shape is [n_feature, n_nodes], weights of neuron.
        random_bias: A Numpy array shape is [n_nodes], bias of neuron.
        beta: A Numpy array shape is [n_feature + n_nodes, n_class], the projection matrix.
        activation: A string of activation name.
        data_std: A list, store normalization parameters for each layer.
        data_mean: A list, store normalization parameters for each layer.
        same_feature: A bool, the true means all the features have same meaning and boundary for example: images.
        task_type: A string of ML task type, 'classification' or 'regression'.
    """

    def __init__(
        self,
        n_nodes,
        lam,
        w_random_vec_range,
        b_random_vec_range,
        activation,
        same_feature=False,
        task_type="classification",
    ):
        assert task_type in [
            "classification",
            "regression",
        ], 'task_type should be "classification" or "regression".'
        self.n_nodes = n_nodes
        self.lam = lam
        self.w_random_range = w_random_vec_range
        self.b_random_range = b_random_vec_range
        self.random_weights = None
        self.random_bias = None
        self.beta = None
        a = Activation()
        self.activation_function = getattr(a, activation)
        self.data_std = None
        self.data_mean = None
        self.same_feature = same_feature
        self.task_type = task_type

    # The FISTA algorithm function is given below

    def fista1(self, A, b, l, maxit):
        J_save = np.zeros((maxit, 1))
        diff = np.zeros((maxit - 1, 1))
        AA = np.dot(A.transpose(), A)
        Lf = np.max(linalg.eigvals(AA))
        Li = 1 / Lf
        alp = l * Li
        m = np.size(A, 1)
        [N, n] = b.shape
        x = np.zeros((m, n))
        yk = x
        tk = 1
        L1 = 2 * Li * AA

        KK = np.dot(np.transpose(A), b)
        L2 = 2 * Li * KK

        # tk1 = 0.5 + 0.5*sqrt(1+4*tk^2);
        for _ in range(0, maxit):
            med = np.dot(L1, yk)
            ck = yk - med + L2
            x1 = (np.maximum(abs(ck) - alp, 0)) * np.sign(ck)

            sqrt = math.sqrt(1 + 4 * tk**2)
            tk1 = 0.5 + 0.5 * sqrt
            tt = (tk - 1) / tk1
            yk = x1 + tt * (x - x1)
            tk = tk1

            x = x1

        return x

    def train(self, data, label, n_class):
        """

        :param data: Training data.
        :param label: Training label.
        :param n_class: An integer of number of class. In regression, this parameter won't be used.
        :return: No return
        """

        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        data = self.standardize(data)  # Normalization data
        n_sample = len(data)
        n_feature = len(data[0])

        self.random_weights = self.get_random_vectors(
            n_feature, self.n_nodes, self.w_random_range
        )
        self.random_bias = self.get_random_vectors(1, self.n_nodes, self.b_random_range)

        h = self.activation_function(
            np.dot(data, self.random_weights)
            + np.dot(np.ones([n_sample, 1]), self.random_bias)
        )

        #########################################################################################################
        rbeta = self.fista1(h, data, 0.01, 100)
        rbeta = np.transpose(rbeta)  # l1-regulraization
        rbias = np.sum(rbeta, axis=0) / self.n_nodes

        rbias = rbias.reshape(-1, self.n_nodes)

        xxx = np.dot(data, rbeta)
        yyy = np.dot(np.ones([n_sample, 1]), rbias)
        h1 = xxx + yyy
        h = h1

        #########################################################################################################

        d = np.concatenate([h, data], axis=1)

        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)

        if self.task_type == "classification":
            y = self.one_hot(label, n_class)
        else:
            y = label
        if n_sample > (self.n_nodes + n_feature):
            self.beta = (
                np.linalg.inv((self.lam * np.identity(d.shape[1]) + np.dot(d.T, d)))
                .dot(d.T)
                .dot(y)
            )
        else:
            self.beta = d.T.dot(
                np.linalg.inv(self.lam * np.identity(n_sample) + np.dot(d, d.T))
            ).dot(y)

    def predict(self, data):
        """

        :param data: Predict data.
        :return: When classification, return Prediction result and probability.
                 When regression, return the output of rvfl.
        """
        data = self.standardize(data)  # Normalization data
        h = self.activation_function(
            np.dot(data, self.random_weights) + self.random_bias
        )
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == "classification":
            proba = self.softmax(output)
            result = np.argmax(proba, axis=1)
            return result, proba
        elif self.task_type == "regression":
            return output

    def eval(self, data, label):
        """

        :param data: Evaluation data.
        :param label: Evaluation label.
        :return: When classification return accuracy.
                 When regression return MAE.
        """

        assert len(data.shape) > 1, "Data shape should be [n, dim]."
        assert len(data) == len(label), "Label number does not match data number."
        assert len(label.shape) == 1, "Label should be 1-D array."

        data = self.standardize(data)  # Normalization data
        h = self.activation_function(
            np.dot(data, self.random_weights) + self.random_bias
        )
        d = np.concatenate([h, data], axis=1)
        d = np.concatenate([d, np.ones_like(d[:, 0:1])], axis=1)
        output = np.dot(d, self.beta)
        if self.task_type == "classification":
            result = np.argmax(output, axis=1)
            acc = np.sum(np.equal(result, label)) / len(label)
            return acc
        elif self.task_type == "regression":
            mae = np.mean(np.abs(output - label))
            return mae

    @staticmethod
    def get_random_vectors(m, n, scale_range):
        x = (scale_range[1] - scale_range[0]) * np.random.random([m, n]) + scale_range[
            0
        ]
        return x

    @staticmethod
    def one_hot(x, n_class):
        y = np.zeros([len(x), n_class])
        for i in range(len(x)):
            y[i, x[i]] = 1
        return y

    def standardize(self, x):
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

    @staticmethod
    def softmax(x):
        return np.exp(x) / np.repeat(
            (np.sum(np.exp(x), axis=1))[:, np.newaxis], len(x[0]), axis=1
        )
