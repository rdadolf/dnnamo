import numpy as np
from sklearn.linear_model import LinearRegression

from ..core.estimator import SKLRegressionEstimator
from ..core.primop import PrimopTypes

class OLSEstimator(SKLRegressionEstimator):
  def __init__(self):
    self._est = { op:LinearRegression() for op in PrimopTypes }

  @property
  def _estimators(self):
    return self._est
