import tensorflow as tf

from ...core.exemplar import Exemplar, ExemplarRegistry
from ...core.primop import PrimopTypes

class TFExemplarRegistry(ExemplarRegistry): pass

class TFExemplar(Exemplar):
  # For common support functions
  pass

################################################################################

class TFExemplar_zero(TFExemplar):
  def synthesize(cls, primop_args):
    dim, = primop_args

class TFExemplar_hadamard(TFExemplar):
  def synthesize(cls, primop_args):
    pass

class TFExemplar_dot(TFExemplar):
  def synthesize(cls, primop_args):
    pass

class TFExemplar_convolution(TFExemplar):
  def synthesize(cls, primop_args):
    pass


################################################################################

TFExemplarRegistry.register('zero', TFExemplar_zero)
TFExemplarRegistry.register('hadamard',TFExemplar_hadamard)
TFExemplarRegistry.register('dot', TFExemplar_dot)
TFExemplarRegistry.register('convolution', TFExemplar_convolution)
