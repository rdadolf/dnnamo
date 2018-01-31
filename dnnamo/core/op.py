from abc import ABCMeta, abstractmethod, abstractproperty

from .identifier import OP

class DnnamoOp(object):
  __metaclass__ = ABCMeta

  def __init__(self, id, optype, parameters, root=None):
    '''Create a Dnnamo Operation.

    Arguments:
      id: a unique string used to identify this operation.
      optype: a string used to identify the generic type of this operation.
      parameters: a list of (name,value) tuples containing operation arguments.
      root: [optional] the original root operation used to create this operation.'''
    self._id = OP(id)
    self._optype = optype
    self._pnames = [k for k,_ in parameters]
    self._pvalues = [v for _,v in parameters]
    self._root = root

  @property
  def id(self):
    'A unique identifier for this operation. Returns an OP identifier object.'
    return self._id

  @property
  def optype(self):
    'A generic identifier which specifies the function of this operation.'
    return self._optype

  @property
  def parameter_names(self):
    'An ordered list of the names of input parameters.'
    return self._pnames

  @property
  def parameter_values(self):
    'An ordered list of the values of input parameters.'
    return self._pvalues

  @property
  def parameters(self):
    'A dictionary of the names and values of input parameters.'
    return {k:v for k,v in zip(self._pnames, self._pvalues)}

  @property
  def root(self):
    'A reference to the native operation this operation shadows.'
    return self._root

  def __str__(self):
    return '<Op_'+str(self.optype)+':'+str(self.id.s)+'>'
