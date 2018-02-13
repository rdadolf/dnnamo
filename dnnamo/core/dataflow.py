from abc import ABCMeta, abstractmethod, abstractproperty
from .identifier import ID

class DnnamoDFO(object):
  __metaclass__ = ABCMeta

  def __init__(self, id, root=None):
    '''Create a Dnnamo dataflow object.

    Arguments 
      id: a unique identifier string.
      root: [optional] the original root object used to create this.'''
    if isinstance(id, ID):
      self._id = id
    else:
      self._id = ID(id)
    self._root = root

  @property
  def id(self):
    return self._id

  @property
  def root(self):
    'A reference to the native object used to create this.'
    return self._root

  @abstractmethod
  def __str__(self):
    return '<DnnamoDFO '+str(self.id)+'>'


class DnnamoEdge(DnnamoDFO):
  __metaclass__ = ABCMeta

  def __init__(self, id, srcs, dsts, root=None):
    '''Create a dataflow edge object.

    Arguments:
      id: a unique identifier string.
      srcs: a list of identifiers for dataflow objects that can produce this.
      dsts: a list of identifiers for dataflow objects that consume this.
      root: [optional] the original root object used to create this.'''
    super(DnnamoEdge,self).__init__(id,root)
    self._srcs = srcs
    self._dsts = dsts

  @property
  def srcs(self):
    'A list of all dataflow objects which can produce this.'
    return self._srcs

  @property
  def dsts(self):
    'A list of all dataflow objects which consume this.'
    return self._dsts

class DnnamoVertex(DnnamoDFO):
  __metaclass__ = ABCMeta

  def __init__(self, id, type, root=None):
    super(DnnamoVertex,self).__init__(id,root)
    self._type = type
  
  @property
  def type(self):
    'A string which specifies a generic function.'
    return self._type


class DnnamoTensor(DnnamoEdge):
  def __init__(self, id, shape, srcs, dsts, root=None):
    '''Create a Dnnamo Tensor.

    Arguments:
      id: a unique string used to identify this tensor.
      shape: a list of integers which defines the dimensions of the tensor.
      srcs: a list of identifiers for operations that can produce this tensor.
      dsts: a list of identifiers for operations that consume this tensor.
      root: [optional] the original root tensor used to create this tensor.'''
    super(DnnamoTensor,self).__init__(id, srcs, dsts, root)
    self.set_shape(shape)

  def set_shape(self, shape):
    if shape is None:
      self._shape = None
    else:
      checked_shape = [int(_) for _ in shape]
      if (checked_shape!=shape):
        raise TypeError('Shape must be either None or a list of zero or more integers, not '+str(shape))
      self._shape = checked_shape 

  @property
  def shape(self):
    '''A Python list of the integer dimensions of this tensor.

    Dimensions listed as -1 are unknown.
    A tensor will always have len(shape)>0.'''
    return self._shape

  # FIXME: Support type information?

  def __str__(self):
    if self.shape:
      return '<T_'+str(self.id)+' ['+','.join([str(d) for d in self.shape])+']>'
    else:
      return '<T_'+str(self.id)+' ?>'

class DnnamoDependence(DnnamoEdge):
  def __str__(self):
    return '<Dep_'+str(self.id)+'>'


class DnnamoOp(DnnamoVertex):
  __metaclass__ = ABCMeta

  def __init__(self, id, type, parameters, root=None):
    '''Create a Dnnamo Operation.

    Arguments:
      id: a unique string used to identify this operation.
      type: a string used to identify the generic type of this operation.
      parameters: a list of (name,value) tuples containing operation arguments.
      root: [optional] the original root operation used to create this operation.'''
    super(DnnamoOp,self).__init__(id,type,root)
    self.set_parameters(parameters)

  def set_parameters(self, parameters):
    self._pnames = [k for k,_ in parameters]
    self._pvalues = [v for _,v in parameters]

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
    return {k:v for k,v in zip(self.parameter_names, self.parameter_values)}

  def __str__(self):
    return '<Op_'+str(self.id)+' '+str(self.type)+'>'

class DnnamoProxy(DnnamoVertex):
  def __str__(self):
    return '<Proxy_'+str(self.id)+' '+str(self.type)+'>'
