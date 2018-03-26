from abc import ABCMeta, abstractmethod
import json

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
