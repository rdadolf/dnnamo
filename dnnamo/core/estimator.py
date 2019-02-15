from abc import ABCMeta, abstractmethod, abstractproperty
import numpy as np

from .registry import Registry

class EstimatorRegistry(Registry):
  '''Registry for holding all Estimator classes.

  Often useful for parsing command-line arguments for tools.'''

################################################################################

class DnnamoEstimator(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def get_params(self):
    'Returns an ordered list of values which represent the estimator parameters.'

  @abstractmethod
  def set_params(self, params):
    'Sets the estimator parameters using an ordered list of values.'

  @abstractmethod
  def estimate(self, op_arguments): pass

  @abstractmethod
  def read(self, filename):
    pass

  @abstractmethod
  def write(self, filename):
    pass

class AnalyticalEstimator(DnnamoEstimator):
  __metaclass__ = ABCMeta

class RegressionEstimator(DnnamoEstimator):
  __metaclass__ = ABCMeta
  @abstractmethod
  def fit(self, features):
    '''Fit the Estimator using data in a DnnamoFeatures object.'''

class SKLRegressionEstimator(RegressionEstimator):
  '''A regression estimator using a scikit-learn estimator.'''
  # scikit-learn is convenient in that its interface is incredibly consistent.
  # As a result, we can re-use a lot of the same functions, regardless of the
  # underlying model. We use an internal property to access these models.

  __metaclass__ = ABCMeta

  @abstractproperty
  def _estimator(self):
    '''The scikit-learn model used to fit and predict values.'''

  def write(self, filename):
    params = np.array(self.get_params())
    with open(filename,'wb') as f:
      np.savez(f, params=params)
      f.flush()

  def read(self, filename):
    npz_filemap = np.load(filename)
    try:
      params = npz_filemap['params']
    except KeyError:
      raise IOError('No estimator parameters found in file "'+str(filename)+'". Is this a Dnnamo Estimator file?')
    self.set_params( params )

  def estimate(self, op_arguments):
    return self._estimator.predict([op_arguments])[0]

  def fit(self, features):
    # In their infinite wisdom, SKL allows multi-target regression with
    # a single dimension.... So passing a y with shape (100,1) causes
    # radically different behavior than passing a y with shape (100,).
    # Retrieving parameters becomes a PITA, so we require a 1-D target.
    ys = np.array(features.measurements)
    if len(ys.shape)!=1:
      raise TypeError, 'Estimator should be fit against a 1-dimensional array, not '+str(ys.shape)

    xs = np.array(features.op_arguments)
    if len(xs.shape)!=2:
      raise TypeError, 'Estimator should be fit using a 2-dimensional array of arguments, not '+str(xs.shape)

    self._estimator.fit(xs, ys)

################################################################################

class EstimatorGroup(object):
  '''Holds a group of DnnamoEstimator objects, one for each primop.'''
  pass
