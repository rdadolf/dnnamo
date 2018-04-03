import numpy as np
from sklearn.linear_model import LinearRegression

from ..core.estimator import SKLRegressionEstimator
from ..core.primop import PrimopTypes

class SKLLinearEstimator(SKLRegressionEstimator):
  pass

  def get_params(self, op):
    coef = self._estimators[op].coef_.tolist()
    intercept = self._estimators[op].intercept_
    return coef+[intercept]

  def set_params(self, op, params):
    self._estimators[op].coef_ = np.array(params[0:-1])
    self._estimators[op].intercept_ = params[-1]

class OLSEstimator(SKLLinearEstimator):
  def __init__(self):
    self._est = { op:LinearRegression() for op in PrimopTypes }

  @property
  def _estimators(self):
    return self._est
