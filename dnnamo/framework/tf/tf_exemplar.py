import tensorflow as tf

from ...core.exemplar import Exemplar, ExemplarRegistry
from ...core.primop import PrimopTypes

class TFExemplarRegistry(ExemplarRegistry): pass

class TFExemplar(Exemplar):
  # For common support functions
  pass

################################################################################

class TFExemplar_zero(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self):
    pass # FIXME

class TFExemplar_hadamard(TFExemplar):
  def __init__(self, primop_args):
    self.dim, = primop_args
  def synthesize(cls):
    pass # FIXME

class TFExemplar_dot(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self):
    pass # FIXME

class TFExemplar_convolution(TFExemplar):
  def __init__(self, primop_args):
    pass # FIXME
  def synthesize(self):
    pass # FIXME


################################################################################

TFExemplarRegistry.register('zero', TFExemplar_zero)
TFExemplarRegistry.register('hadamard',TFExemplar_hadamard)
TFExemplarRegistry.register('dot', TFExemplar_dot)
TFExemplarRegistry.register('convolution', TFExemplar_convolution)
