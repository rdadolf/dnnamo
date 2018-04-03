from abc import ABCMeta, abstractmethod, abstractproperty
import json
import numpy as np
import sklearn

from .primop import PrimopTypes

class EstimatorIO(dict):
  # EstimatorIO just contains a dictionary of op:lists, with data in the form:
  # { op1: [  ( [arg0,arg1,arg2,...], measurement ),
  #           ( [arg0,arg1,arg2,...], measurement ),
  #           ... ],
  #   op2: [  ( [arg0,...], measurement ),
  #           ( [arg0,...], measurement ),
  #           ... ]
  #   ... }

  def __init__(self, *args, **kwargs):
    super(EstimatorIO, self).__init__(*args, **kwargs)
    self._validate_keys()
    for t in PrimopTypes:
      if t not in self:
        self[t] = []
    self._validate_data()

  def _validate_keys(self):
    for t in self.keys():
      if type(t)!=str:
        raise TypeError, 'Operator names must be strings, not '+str(type(t))
      if t not in PrimopTypes:
        raise KeyError, 'Unrecognized operation type: '+str(t)

  def _validate_data(self):
    for k,v in self.items():
      try:
        len(v)
      except TypeError as e:
        raise e, 'operator data must be a list of argument-measurement pairs, not '+str(type(v))
      for pair in v:
        if len(pair)!=2:
          raise IndexError, 'Operator data elements must be argument-measurement pairs, not '+str(type(pair))+' with length '+str(len(pair))
        try:
          len(pair[0])
        except TypeError as e:
          raise e, 'First element of pair must be an operator argument list, not '+str(type(pair[0]))

  def append(self, op, pair):
    self[op].append(pair)

  def extend(self, op, pair_list):
    self[op].extend(pair)

  def read(self, filename):
    with open(filename, 'r') as f:
      self.__init__(json.load(f))

  def write(self, filename):
    with open(filename, 'w') as f:
      json.dump(self, f)

################################################################################

class DnnamoEstimator(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_params(self, op):
    'Returns an ordered list of values which represent the estimator parameters.'

  @abstractmethod
  def set_params(self, op, params):
    'Sets the estimator parameters for an op using an ordered list of values.'

  @abstractmethod
  def estimate(self, op, op_arguments): pass

class AnalyticalEstimator(DnnamoEstimator):
  __metaclass__ = ABCMeta

class RegressionEstimator(DnnamoEstimator):
  __metaclass__ = ABCMeta
  @abstractmethod
  def fit(self, op, op_argument_list, measurement_list): pass

class SKLRegressionEstimator(RegressionEstimator):
  '''A regression estimator using a scikit-learn estimator.'''
  # scikit-learn is convenient in that its interface is incredibly consistent.
  # As a result, we can re-use a lot of the same functions, regardless of the
  # underlying model. We use an internal property to access these models.

  __metaclass__ = ABCMeta

  @abstractproperty
  def _estimators(self):
    '''Returns a dictionary of sk-learn estimator objects, keyed by op type.'''

  # DnnamoEstimators use ordinal parameters, sk-learn uses k-v parameters.
  # To work around this, we assume sk-learn model parameters are static (which
  # they all are) and simply use the sorted keys as the canonical ordering.

  #def get_params(self, op):
  #  kv = self._estimators[op].get_params(deep=True)
  #  key_order = sorted(kv.keys())
  #  return [kv[k] for k in key_order]

  #def set_params(self, op, params):
  #  current_kv = self._estimators[op].get_params(deep=True)
  #  key_order = sorted(current_kv.keys())
  #  new_kv = dict( zip(key_order, params) )
  #  self._estimators[op].set_params(**new_kv)

  def estimate(self, op, op_arguments):
    return self._estimators[op].predict([op_arguments])[0]

  def fit(self, op, op_argument_list, measurement_list):
    # In their infinite wisdom, SKL allows multi-target regression with
    # a single dimension.... So passing a y with shape (100,1) causes
    # radically different behavior than passing a y with shape (100,).
    # Retrieving parameters becomes a PITA, so we require a 1-D target.
    ys = np.array(measurement_list)
    if len(ys.shape)!=1:
      raise TypeError, 'Estimator should be fit against a 1-dimensional array, not '+str(ys.shape)

    xs = np.array(op_argument_list)
    if len(xs.shape)!=2:
      raise TypeError, 'Estimator should be fit using a 2-dimensional array of arguments, not '+str(xs.shape)

    self._estimators[op].fit(xs, ys)
