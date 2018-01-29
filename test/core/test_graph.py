import unittest

from dnnamo.core.op import DnnamoOp
from dnnamo.core.tensor import DnnamoTensor
from dnnamo.core.graph import DnnamoGraph

from .synthetic_graph import SynthOp, SynthTensor, ConstructRandomGraph

class TestGenericGraph(unittest.TestCase):
  def test_instantiation(self):
    g = DnnamoGraph()

  def test_iteration(self):
    N_OPS = 7
    N_TS = 10
    with ConstructRandomGraph(N_OPS, N_TS) as g:

      assert len(g.ops)==N_OPS
      for op in g.ops:
        assert op.optype=='Synth', 'Invalid operation found.'
        assert isinstance(op, SynthOp), 'Corrupted tensor object: '+str(op)

      assert len(g.tensors)==N_TS
      for t in g.tensors:
        assert isinstance(t, SynthTensor), 'Corrupted tensor object: '+str(t)
