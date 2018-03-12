from abc import ABCMeta, abstractmethod

import numpy as np

class ArgSampler(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def sample(self, primop_type, n=1):
    pass

################################################################################

class UniformSampler(object):
  min_T_dimsize = 10
  max_T_dimsize = 1000
  min_T_dims = 1
  max_T_dims = 4

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
    dims = np.random.random_integers(self.min_T_dims, self.max_T_dims, size=n)
    sizes = [ [ np.random.random_integers(self.min_T_dimsize, self.max_T_dimsize, size=d).tolist() ] for d in dims ]
    return sizes

  #def sample_dot(self, n): pass
  #def sample_convolution(self, n): pass

################################################################################

# FIXME: Non-uniform sampler
