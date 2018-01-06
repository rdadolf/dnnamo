from abc import ABCMeta, abstractmethod

class Translator(object):
  '''A graph pattern-matching and rewrite engine.'''
  __metaclass__ = ABCMeta

  @abstractmethod
  def translate(self, native_graph): pass

  @abstractmethod
  def map_native_op(self, native_op_id): pass

  @abstractmethod
  def map_primop(self, primop_id): pass
