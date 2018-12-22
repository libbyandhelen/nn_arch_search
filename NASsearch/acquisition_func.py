""" module for acquisition function"""
import numpy as np
from scipy.stats import norm

from Kernel_optimization.all_gp import GaussianProcess

class AcquisitionFunc:
    """ class for acquisition function
    expected improvement in this case
    """
    def __init__(self, X_train, y_train, current_optimal, mode, trade_off):
        """
        :param mode: pi: probability of improvement, ei: expected improvement, lcb: lower confident bound
        :param trade_off: a parameter to control the trade off between exploiting and exploring
        :param model_type: gp: gaussian process, rf: random forest
        """
        self.X_train = X_train
        self.y_train = y_train
        self.current_optimal = current_optimal
        self.mode = mode or "ei"
        self.trade_off = trade_off or 0.1
        self.model = GaussianProcess(80)

    def compute(self, X_test, weight_file):
        y_means, y_vars, y_stds = self.model.fit_predict(self.X_train, X_test, self.y_train, weight_file)
        z = (y_means - self.current_optimal - self.trade_off) / y_stds

        if self.mode == "ei":
            result = y_stds * (z * norm.cdf(z) + norm.pdf(z))
        elif self.mode == "pi":
            result = norm.cdf(z)
        elif self.mode == "ucb":
            result = y_means + self.trade_off * y_stds
        else:
            result = - (y_means - self.trade_off * y_stds)
        return result, y_means, y_stds

