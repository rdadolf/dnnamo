from abc import ABCMeta, abstractmethod, abstractproperty

class DnnamoTensor(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self, shape, srcs, dsts, root=None): pass # FIXME: more arguments here, probably

  @abstractproperty
  def id(self):
    'A unique identifier for this operation. Returns a T identifier object.'

  @abstractproperty
  def shape(self):
    '''A Python list of the integer dimensions of this tensor.

    Dimensions listed as -1 are unknown.
    A tensor will always have len(shape)>0.'''

  # FIXME: Support type information

  @abstractproperty
  def srcs(self):
    'A list of all operations which can produce this tensor.'

  @abstractproperty
  def dsts(self):
    'A list of all operations which use this tensor as input.'

  @abstractproperty
  def root(self):
    'A reference to the native tensor that this operation shadows.'

  def __str__(self):
    return '<T_'+str(self.id.s)+':'+','.join([str(d) for d in self.shape])+'>'
