"""Example of rolling out an sklearn-like estimator"""
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import euclidean_distances


class TemplateClassifier(BaseEstimator, ClassifierMixin):
    """Simple nearest-neighbor classifier

    References:
        `Rolling your own estimator <goo.gl/Rk62Ql>`_
    """
    def __init__(self, demo_param='demo'):
        self.demo_param = demo_param
        self.xtrain_ = None
        self.ytrain_ = None
        self.classes_ = None

    def fit(self, xtrain, ytrain):
        """Fit method"""
        # Check that X and y have correct shape
        xtrain, ytrain = check_X_y(xtrain, ytrain)
        # Store the classes seen during fit
        self.classes_ = unique_labels(ytrain)

        self.xtrain_ = xtrain
        self.ytrain_ = ytrain
        # Return the classifier
        return self

    def predict(self, xtest):
        """Predict method"""
        # Check is fit had been called
        check_is_fitted(self, ['xtrain_', 'ytrain_'])

        # Input validation
        xtest = check_array(xtest)

        closest = np.argmin(euclidean_distances(xtest, self.xtrain_), axis=1)
        return self.ytrain_[closest]
