#!/usr/bin/env python
# encoding=utf8

import math

import numpy as np

from ml_metrics import auc


class BasicLogisticRegression(object):
    """
    the most basic logistic regression model, containing intercept, without weighting samples

    p(y = 1|w,x) = 1 / (1 + exp(-w*x))
    p(y = 0|w,x) = 1 - p(y = 1|w,x)

    data likelihood = \prod p^y * (1-p)^(1-y)
    we can get the object function as the negative log likelihood:
    - \sum (y * ln(p) + (1 - y) * ln(1 - p))

    hence we can get the gradient of the object function as:
    (note: \frac{\partial p} {\partial w} = p * (1-p) * x)
    \sum (p - y) * x
    """

    dimension = -1
    alpha = 0.1
    beta = 0.0
    sample_number = 0
    weights = None

    def __init__(self, d, alpha=0, beta=0):
        """
        :param d: dimension of input vector, intercept is not included
        :param alpha: weight of L1 regularization
        :param beta: weight of L2 regularization
        """
        print("init LR model with dimension=%d" % d)
        self.dimension = d
        self.alpha = alpha
        self.beta = beta

        self.sample_number = 0
        self.weights = np.zeros(d + 1)

    def get_dimension(self):
        return self.dimension

    def _pred_raw(self, margin):
        return 1 / (1 + math.exp(-margin))

    def _pred_raw_np(self, x):
        assert (len(x) + 1) == len(self.weights)
        with_intercept = np.insert(x, 0, 1.0)
        margin = np.dot(self.weights, with_intercept)
        return self._pred_raw(margin)

    def _pred_raw_dict(self, x):
        # margin = self.weights[0]
        margin = 0

        for i, v in x.items():
            margin += self.weights[i] * v
            return self._pred_raw(margin)

    def _pred_raw_one_indices(self, x):
        margin = self.weights[0] + sum(map(lambda i: self.weights[i], x))
        return self._pred_raw(margin)

    def predict_raw(self, x):
        """
        y = 1 / (1 + exp(-w*x))
        """
        # print x
        if type(x) is np.array:
            return self._pred_raw_np(x)
        elif type(x) is dict:
            return self._pred_raw_dict(x)
        elif hasattr(x, '__iter__'):
            return self._pred_raw_one_indices(x)
        else:
            raise Exception('unknown input type:' + type(x))

    def sgd_fit_one(self, x, y):
        self.sample_number += 1
        step = 1 / math.sqrt(self.sample_number + self.dimension)
        # step = 0.001
        # todo: L1 regularization
        self.weights *= (1 - step * self.beta)
        if type(x) is np.array or type(x) is npz.ndarray:
            self.weights -= (self._pred_raw_np(x) - y) * x
        elif type(x) is dict:
            # print "..."
            gap = step * (self._pred_raw_dict(x) - y)
            # self.weights[0] -= gap
            for i, v in x.items():
                self.weights[i] -= gap * v

        # theta + a*(y[i]-h(x[i],theta))*x[i]
        elif hasattr(x, '__iter__'):
            gap = step * (self._pred_raw_one_indices(x) - y)
            self.weights[0] -= gap
            for i in x:
                self.weights[i] -= gap
        else:
            raise Exception('unknown input type:' + type(x))

        if self.sample_number % 100000 == 1:
            print "count=%d\tintercept=%f" % (self.sample_number, self.weights[0])
            # print "weights:"+', '.join(["%d:%f" % (i, v) for i, v in enumerate(self.weights) if v != 0.0])

    def object_func_bacth(self, dataset):
        """
        object function value = - \sum (y * ln(p) + (1 - y) * ln(1 - p))
        gradient = \sum (p - y) * x
        """
        pass

    def __str__(self):
        rst = [self.dimension, self.sample_number]
        rst.extend(self.weights)
        return '\n'.join(map(str, rst))


def first_test():
    from ml_metrics import auc
    import random
    from sklearn import datasets

    b = BasicLogisticRegression(4)

    iris = datasets.load_iris()
    train_data = iris.data[:75]
    train_y = iris.target[:75]

    test_x = iris.data[75:100]
    tmp = iris.target[:100]
    random.shuffle(tmp)
    test_y = tmp[:50]

    def to_dict(x):
        return {i: k for i, k in enumerate(x, start=1)}

    for z in xrange(50):
        for x, y in random.shuffle(zip(train_data, train_y)):
            # print x, y
            b.sgd_fit_one(to_dict(x), y)
    print "fit done"

    rst_y = map(b.predict_raw, map(to_dict, test_x))
    print b.weights
    print test_y
    print rst_y
    print auc(test_y, rst_y)
    # print len(iris.data)
    #



    # another implementation
    from sgd import log_reg_sgd, h

    theta, err = log_reg_sgd(train_data, train_y, 0.001, max_iter=100)
    pred = [h(i, theta) for i in test_x]
    print "theta,", theta
    print auc(test_y, pred)


if __name__ == '__main__':
    from sklearn import datasets

    EX = 500

    # learning rate
    a = 0.001
    max_iter = 10
    # create a synthetic data set
    x, y = datasets.make_classification(EX)
    print "sample", x[251]
    print "feature num ", x.shape[1]
    # append a 1 column at index 0 in x
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    print x[251]
    from sgd import log_reg_sgd, h


    theta = log_reg_sgd(x[:EX / 2], y[:EX / 2], a, max_iter=max_iter)
    pred = [h(x[i], theta) for i in xrange(EX / 2, EX)]
    print "weights ",theta
    # print "err ",err
    print auc(y[EX / 2:], pred)


    def to_dict(x):
        # print x
        return {i: k for i, k in enumerate(x[1:], start=1)}


    b = BasicLogisticRegression(x.shape[1]-1, a)
    for z in xrange(max_iter ):
        for i in xrange(EX / 2):
            b.sgd_fit_one(to_dict(x[i]), y[i])

    rst_y = map(b.predict_raw, map(to_dict, x[EX / 2:]))
    print rst_y
    print b.weights
    print auc(y[EX / 2:], rst_y)
