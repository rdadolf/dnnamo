from abc import ABCMeta, abstractmethod
from .registry import Registry

class Exemplar(object):
  '''An exemplar is a native operation chosen to represent the performance of a primop.

  Each primop groups the performance of several different native ops together.
  Primop performance cannot be measured directly, however, so in order to model
  and measure the performance of a primop across a variety of inputs, there
  needs to be a native operation that can be synthesized and run to collect
  data.'''

  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self, primop_argvalues):
    pass

  @abstractmethod
  def synthesize(self, *args, **kwargs):
    pass


class ExemplarRegistry(Registry):
  '''Registry mapping primops to exemplars.

  Note: This class will never contain anything. It should always be subclassed
  by a framework to actually implement its exemplar classes.'''
