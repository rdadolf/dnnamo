from abc import ABCMeta, abstractproperty
from collections import namedtuple
import tensorflow as tf

from ...core.exemplar import Exemplar, ExemplarRegistry

class TFExemplarRegistry(ExemplarRegistry):
  pass # Fully defined from superclass; this allocates distinct storage for it.

class TFExemplar(Exemplar):
  __metaclass__ = ABCMeta

  # For common support functions
  @abstractproperty
  def input_signature(self): pass
  @abstractproperty
  def output_signature(self): pass

  def get_op_name(self):
    return 'Exemplar'


################################################################################


class TFSignatureType(object):
  Tensor = namedtuple('Tensor',['dtype','dims'])
  Scalar = namedtuple('Scalar',['dtype'])
  types = [Tensor, Scalar]

################################################################################

class TFExemplar_zero(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self): # pylint: disable=W0221
    pass # FIXME
  @property
  def input_signature(self):
    pass # FIXME
  @property
  def output_signature(self):
    pass # FIXME

class TFExemplar_hadamard(TFExemplar):
  def __init__(self, primop_argvalues):
    self.dims = primop_argvalues

  @property
  def input_signature(self):
    return [
      TFSignatureType.Tensor(dtype='float32', dims=self.dims),
      TFSignatureType.Tensor(dtype='float32', dims=self.dims)
    ]

  @property
  def output_signature(self):
    return [
      TFSignatureType.Tensor(dtype='float32', dims=self.dims)
    ]

  def synthesize(self, inputs): # pylint: disable=W0221
    lhs,rhs = inputs
    #outT = tf.add(lhs,rhs, name='Exemplar')
    outT = tf.multiply(lhs,rhs, name='Exemplar')
    return [outT]

class TFExemplar_dot(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self): # pylint: disable=W0221
    pass # FIXME
  @property
  def input_signature(self):
    pass # FIXME
  @property
  def output_signature(self):
    pass # FIXME

class TFExemplar_convolution(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self): # pylint: disable=W0221
    pass # FIXME
  @property
  def input_signature(self):
    pass # FIXME
  @property
  def output_signature(self):
    pass # FIXME


################################################################################

#TFExemplarRegistry.register('zero', TFExemplar_zero)
TFExemplarRegistry.register('hadamard',TFExemplar_hadamard)
#TFExemplarRegistry.register('dot', TFExemplar_dot)
#TFExemplarRegistry.register('convolution', TFExemplar_convolution)
