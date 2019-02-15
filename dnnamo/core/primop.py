from abc import ABCMeta

from .dataflow import DnnamoOp
from .identifier import ID
from .registry import Registry

# This is just the interface definition.
class Primop(DnnamoOp):
  __metaclass__ = ABCMeta

  def __init__(self, arguments=None, root=None):
    self._id = self._unique_id()
    if arguments is not None:
      assert len(arguments)==len(self.argnames), 'Incompatible argument list: '+str(len(arguments))+' given, '+str(len(self.argnames))+' expected'
      self._argvalues = [_ for _ in arguments] # copy
    else:
      self._argvalues = [None for _ in self.argnames]
    self._root = root

  # Class-wide counter
  _id_counter = 0
  # Instance function tracking class-wide counter
  def _unique_id(self):
    '''Returns an int guaranteed to be unique across all Primop subclasses.'''
    return ID.unique(self.type)

  # Factory-assigned properties:
  # type
  # argnames

  @property
  def id(self):
    return self._id

  @property
  def root(self):
    return self._root

  def __str__(self):
    return '<Primop_'+str(self.id)+' '+str(self.type)+'>'

class PrimopTypes(Registry):
  '''Singleton container class for all primitive operation types.'''

  @staticmethod
  def _declare(t, argument_set, desc=None):
    '''Internal shortcut for dynamically creating new Primop class types.'''
    primop_typename = 'Primop_'+str(t)
    # Create factory-assigned properties
    def type_prop(self): return t
    def argument_names_property(self): return [_ for _ in argument_set] # copy
    if desc is None:
      desc = 'Dnnamo primitive operation.'
    # Create new type
    NewPrimop = type(primop_typename, (Primop,), {
      'type': property(type_prop),
      'argnames': property(argument_names_property),
      '__doc__': desc,
    })
    PrimopTypes.register(t,NewPrimop)
    return NewPrimop # Return new type to get its name assigned


################################################################################
# Primop definitions

# NOTE: All Primop argument lists are flattened, so each argument should be a
#       single value. This not true of native operations.
# NOTE: Primops are currently limited to 4-dimensional tensors.

# This is a primop for undefined operations.
# It's valid in dependence graphs, but will obviously give unrealistic values if
# it is used for modeling. Still, it's useful to let us purposefully under-cover
# the source framework's operation space or intentionally ignore certain cheap
# operations.
Primop_undef = PrimopTypes._declare('undef', [],
  desc='Undefined Primop. Used when the native operation is unknown.')

Primop_zero = PrimopTypes._declare('zero', [],
  desc='A free or zero-cost operation.')

# FIXME: I'm not sure how to implement fixed-cost operations. It might be hard
# to unify them. For instance, let's say there's two native op types A and B.
# On platform X, A costs 1, B costs 2. On platform Y, A costs 3, B costs 2.
# Since primops are supposed to use a single cost model to represent different
# operations, this won't work.
#Primop_fixed = PrimopTypes._declare('fixed', [],
#  desc='A constant-cost operation.')

Primop_hadamard = PrimopTypes._declare('hadamard',
  ['dim0','dim1','dim2','dim3'],
  desc='''A hadamard (element-wise matrix) operation.

  Arguments:
    dim: a python list of integer dimensions of the input and output tensors.''')

Primop_dot = PrimopTypes._declare('dot',
  ['a_dim0','a_dim1','a_dim2','a_dim3',
   'b_dim0','b_dim1','b_dim2','b_dim3',
   'dim_reduce'],
  desc='''Inner (dot) product. This includes N-dimensional matrix multiplication.

  Arguments:
    dim_a: a python list of integer dimensions of the first input tensor.
    dim_b: a python list of integer dimensions of the second input tensor.
    dim_reduce: a single integer dimension the dot product reduces over.''')

Primop_convolution = PrimopTypes._declare('convolution',
  ['a_dim0','a_dim1','a_dim2','a_dim3',
   'b_dim0','b_dim1','b_dim2','b_dim3'],
  desc='''Convolution over a matrix a with a filter b.

  Arguments:
    dim_a: a python list of integer dimensions of the input tensor.
    dim_b: a python list of integer dimensions of the filter.''')

