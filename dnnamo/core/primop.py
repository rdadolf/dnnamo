from abc import ABCMeta, abstractproperty

from .dataflow import DnnamoOp
from .identifier import ID
from .registry import Registry

# This is just the interface definition.
class Primop(DnnamoOp):
  __metaclass__ = ABCMeta

  def __init__(self, parameters=None, root=None):
    self._id = self._unique_id()
    if parameters is not None:
      self._params = {p:parameters[p] for p in self.parameter_names}
    else:
      self._params = {p:None for p in self.parameter_names}
    self._root = root

  # Class-wide counter
  _id_counter = 0
  # Instance function tracking class-wide counter
  def _unique_id(self):
    '''Returns an int guaranteed to be unique across all Primop subclasses.'''
    return ID.unique(self.type)
    #c = Primop._id_counter
    #Primop._id_counter += 1
    #primop_id = str(self.type)+'_'+str(c)
    #return primop_id

  # Factory-assigned properties:
  # type
  # parameter_names

  @property
  def id(self):
    return self._id
  @property
  def parameter_values(self):
    return [self._params[k] for k in self.parameter_names]
  @property
  def root(self):
    return self._root

  def __str__(self):
    return '<Primop_'+str(self.id)+' '+str(self.type)+'>'

class PrimopTypes(Registry):
  '''Singleton container class for all primitive operation types.'''
  #primops = {}

  #@classmethod
  #def items(cls):
  #  return cls.primops.items()

  #@classmethod
  #def __iter__(cls):
  #  for p in cls.primops:
  #    yield p

  #@classmethod
  #def __len__(cls):
  #  return len(cls.primops)

  #@classmethod
  #def __getitem__(cls, key):
  #  return cls.primops[key]

  @staticmethod
  def new(t, parameter_set, desc=None):
    '''Shortcut for dynamically creating new Primop class types.

    In general, this function should not be used outside this file.'''

    primop_typename = 'Primop_'+str(t)
    # Create factory-assigned properties
    def type_prop(self): return t
    def parameter_names_prop(self): return [p for p in parameter_set] # copy
    if desc is None:
      desc = 'Dnnamo primitive operation.'
    # Create new type
    NewPrimop = type(primop_typename, (Primop,), {
      'type': property(type_prop),
      'parameter_names': property(parameter_names_prop),
      '__doc__': desc,
    })
    # Record the new type
    PrimopTypes.register(t,NewPrimop)
    # Return new type to get its name assigned
    return NewPrimop


################################################################################
# Primop definitions

# This is a primop for undefined operations.
# It's valid in dependence graphs, but will obviously give unrealistic values if
# it is used for modeling. Still, it's useful to let us purposefully under-cover
# the source framework's operation space or intentionally ignore certain cheap
# operations.
Primop_undef = PrimopTypes.new('undef', [],
  desc='Undefined Primop. Used when the native operation is unknown.')

Primop_zero = PrimopTypes.new('zero', [],
  desc='A free or zero-cost operation.')

# FIXME: I'm not sure how to implement fixed-cost operations. It might be hard
# to unify them. For instance, let's say there's two native op types A and B.
# On platform X, A costs 1, B costs 2. On platform Y, A costs 3, B costs 2.
# Since primops are supposed to use a single cost model to represent different
# operations, this won't work.
#Primop_fixed = PrimopTypes.new('fixed', [],
#  desc='A constant-cost operation.')

Primop_hadamard = PrimopTypes.new('hadamard', ['dim'],
  desc='''A hadamard (element-wise matrix) operation.

  Parameters:
    dim: a python list of integer dimensions of the input and output tensors.''')

Primop_dot = PrimopTypes.new('dot', ['dim_a', 'dim_b', 'dim_reduce'],
  desc='''Inner (dot) product. This includes N-dimensional matrix multiplication.

  Parameters:
    dim_a: a python list of integer dimensions of the first input tensor.
    dim_b: a python list of integer dimensions of the second input tensor.
    dim_reduce: a single integer dimension the dot product reduces over.''')

Primop_convolution = PrimopTypes.new('convolution', ['dim_a', 'dim_b'],
  desc='''Convolution over a matrix a with a filter b.

  Parameters:
    dim_a: a python list of integer dimensions of the input tensor.
    dim_b: a python list of integer dimensions of the filter.''')

