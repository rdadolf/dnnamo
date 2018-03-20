from abc import ABCMeta, abstractmethod

import numpy as np

class ArgSampler(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def sample(self, primop_type, n=1):
    pass

################################################################################

class UniformArgSampler(object):
  min_T_dimsize = 1
  max_T_dimsize = 100

  def sample(self, primop_type, n=1, seed=None):
    if seed is not None:
      np.random.seed(seed)
    method = getattr(self, 'sample_'+primop_type)
    return method(n)

  def sample_undef(self, n):
    return [ [] for _ in xrange(0,n) ]

  def sample_zero(self, n):
    return [ [] for _ in xrange(0,n) ]

  def sample_hadamard(self, n):
    return [ np.random.random_integers(self.min_T_dimsize, self.max_T_dimsize, size=4).tolist() for _ in xrange(0,n) ]

  #def sample_dot(self, n): pass
  #def sample_convolution(self, n): pass

################################################################################

# FIXME: Non-uniform sampler
