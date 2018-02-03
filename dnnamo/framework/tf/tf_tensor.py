from ...core.identifier import OP
from ...core.tensor import DnnamoTensor

class TFTensor(DnnamoTensor):

  @classmethod
  def _fix_tensorname(cls,tensorname):
    t = tensorname
    # Output tensors in slot 0 can be canonically named without a slot.
    if t.endswith(':0'):
      t = t[:-2]
    return t


  @classmethod
  def from_root_tensor(cls, root_tensor):
    '''Create a TFTensor from a TensorFlow Tensor object.

    Arguments:
      root_op: a TensorFlow Tensor object.'''

    id = cls._fix_tensorname(root_tensor.name)
    if (root_tensor.shape.ndims is None) or (len(root_tensor.shape.as_list())<1):
      shape = []
    else:
      shape = [-1 if s is None else s for s in root_tensor.shape.as_list()]
    srcs = [OP(root_tensor.op.name)]
    dsts = [OP(op.name) for op in root_tensor.consumers()]

    return cls(id=id, shape=shape, srcs=srcs, dsts=dsts, root=root_tensor)
