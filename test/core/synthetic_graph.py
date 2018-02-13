import random

from dnnamo.core.identifier import ID
from dnnamo.core.dataflow import DnnamoTensor, DnnamoOp
from dnnamo.core.graph import DnnamoGraph

class SynthOp(DnnamoOp):
  def __init__(self, parameters=None, root=None):
    self._id = ID.unique('synth')
    self._p = {'foo': 'bar'}
  @property
  def id(self):
    return self._id
  @property
  def type(self):
    return 'Synth'
  @property
  def parameter_names(self):
    return ['foo']
  @property
  def parameter_values(self):
    return [self._p[k] for k in self.parameter_names]
  @property
  def parameters(self):
    return self._p.items()
  @property
  def root(self):
    return None

class SynthTensor(DnnamoTensor):
  pass # FIXME: once DnnamoTensor exists, fill this in
  def __init__(self, shape, srcs, dsts, root=None):
    if len(shape)<1:
      shape = [1]
    self._shape = tuple(shape)
    self._id = ID.unique('synth')
    self._srcs = srcs
    self._dsts = dsts
    self._root = root
  @property
  def id(self):
    return self._id
  @property
  def srcs(self):
    return self._srcs
  @property
  def dsts(self):
    return self._dsts
  @property
  def shape(self):
    return self._shape
  @property
  def root(self):
    return self._root

class ConstructRandomGraph(object):
  def __init__(self, n_ops, n_tensors, seed=13):
    random.seed(seed)
    self._g = DnnamoGraph()
    for _ in xrange(0,n_ops):
      op = SynthOp()
      self._g.add_op(op)
    for _ in xrange(0,n_tensors):
      src = random.choice(list(self._g.ops))
      dst = random.choice(list(self._g.ops))
      t = SynthTensor([1],[src.id],[dst.id])
      self._g.add_tensor(t)
  def __enter__(self):
    return self._g
  def __exit__(self, *args, **kwargs):
    pass

