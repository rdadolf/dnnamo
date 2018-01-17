from abc import ABCMeta, abstractmethod

class BaseLoader(object):
  __metaclass__=ABCMeta

  @abstractmethod
  def __init__(self, identifier): pass

  @abstractmethod
  def load(self): pass

  @property
  def model(self):
    '''Returns a cached DnnamoModel object if already loaded, otherwise loads it.'''
    # This is an optional function, so if it is not overridden, simply load the
    # model again.
    return self.load()

