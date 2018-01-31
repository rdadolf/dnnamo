from .identifier import T,OP

class DnnamoTensor(object):

  def __init__(self, id, shape, srcs, dsts, root=None):
    '''Create a Dnnamo Tensor.

    Arguments:
      id: a unique string used to identify this tensor.
      shape: a list of integers which defines the dimensions of the tensor.
      srcs: a list of OP identifiers for operations that can produce this tensor.
      dsts: a list of OP identifiers for operations that consume this tensor.
      root: [optional] the original root tensor used to create this tensor.'''
    self._id = T(id)
    checked_shape = [int(_) for _ in shape]
    if (checked_shape!=shape):
      raise TypeError('Shape must be a list of zero or more integers, not '+str(shape))
    self._shape = checked_shape 
    bad_srcs = [_ for _ in srcs if not isinstance(_,OP)]
    if len(bad_srcs)>0:
      raise TypeError('srcs must be a list of OP identifiers, not '+','.join(map(str,bad_srcs)))
    self._srcs = srcs
    bad_dsts = [_ for _ in dsts if not isinstance(_,OP)]
    if len(bad_dsts)>0:
      raise TypeError('dsts must be a list of OP identifiers, not '+','.join(map(str,bad_dsts)))
    self._dsts = dsts
    self._root = root

  @property
  def id(self):
    'A unique identifier for this operation. Returns a T identifier object.'
    return self._id

  @property
  def shape(self):
    '''A Python list of the integer dimensions of this tensor.

    Dimensions listed as -1 are unknown.
    A tensor will always have len(shape)>0.'''
    return self._shape

  # FIXME: Support type information?

  @property
  def srcs(self):
    'A list of all operations which can produce this tensor as OP identifiers.'
    return self._srcs

  @property
  def dsts(self):
    'A list of all operations which use this tensor as input as OP identifiers.'
    return self._dsts

  @property
  def root(self):
    'A reference to the native tensor that this operation shadows.'
    return self._root

  def __str__(self):
    return '<T_'+str(self.id.s)+':'+','.join([str(d) for d in self.shape])+'>'
