from abc import ABCMeta, abstractmethod

class DnnamoEstimator(object):
  @abstractmethod
  def get_params(self, op): pass

  @abstractmethod
  def set_params(self, op, *params): pass

  @abstractmethod
  def estimate(self, op, op_arguments): pass

class AnalyticalEstimator(DnnamoEstimator):
  pass

class RegressionEstimator(DnnamoEstimator):

  @abstractmethod
  def fit(self, op, op_argument_list, measurement_list): pass
