import numpy as np
from sklearn.linear_model import LinearRegression

from ..core.estimator import RegressionEstimator
from ..core.primop import PrimopTypes

class OLSEstimator(RegressionEstimator):
  def __init__(self):
    self._est = { op:LinearRegression() for op in PrimopTypes }
  def get_params(self, op):
    raise NotYetImplemented # FIXME
  def set_params(self, op, *params):
    raise NotYetImplemented # FIXME
  def estimate(self, op, op_arguments):
    raise NotYetImplemented # FIXME
  def fit(self, op, op_argument_list, measurement_list):
    raise NotYetImplemented # FIXME

