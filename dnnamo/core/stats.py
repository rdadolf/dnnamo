from abc import ABCMeta, abstractmethod

class NativeStats(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self, model, traces): pass


