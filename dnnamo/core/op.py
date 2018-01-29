from abc import ABCMeta, abstractmethod, abstractproperty

class DnnamoOp(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def __init__(self, parameters=None, root=None): pass

  @abstractproperty
  def id(self):
    'A unique identifier for this operation. Returns an OP identifier object.'

  @abstractproperty
  def optype(self):
    'A generic identifier which specifies the function of this operation.'

  @abstractproperty
  def parameter_names(self):
    'An ordered list of the names of input parameters.'

  @abstractproperty
  def parameter_values(self):
    'An ordered list of the values of input parameters.'

  @abstractproperty
  def parameters(self):
    'A dictionary of the names and values of input parameters.'

  @abstractproperty
  def root(self):
    'A reference to the native operation this operation shadows.'

  def __str__(self):
    return '<Op_'+str(self.optype)+':'+str(self.id.s)+'>'
