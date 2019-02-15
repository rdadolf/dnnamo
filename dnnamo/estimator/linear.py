import numpy as np
from sklearn.linear_model import LinearRegression

from ..core.estimator import SKLRegressionEstimator, EstimatorRegistry

class OLSEstimator(SKLRegressionEstimator):
  def __init__(self):
    self._est = LinearRegression()

  def get_params(self):
    try:
      coef = self._estimator.coef_.tolist()
      intercept = self._estimator.intercept_
    except AttributeError:
      raise ValueError('Estimator has not been fit, so it has no parameters yet.')
    return coef+[intercept]

  def set_params(self, params):
    self._estimator.coef_ = np.array(params[0:-1])
    self._estimator.intercept_ = params[-1]

  @property
  def _estimator(self):
    return self._est

EstimatorRegistry.register('ols', OLSEstimator)
