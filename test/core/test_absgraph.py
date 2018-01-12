import unittest
from .synth import AbstractGraphTestCase

class TestAbstractGraph(AbstractGraphTestCase):
  def test_adjacency_constraints(self):
    G = self.G
    assert len(G)==3, 'Added three primops, graph is not three nodes'
    (p0,p1,p2) = G.primops.values()
    # Check simple edge invariants
    deps_p0 = G.dep(p0)
    assert len(deps_p0)==1 and (deps_p0!=p0), 'Corrupted adjacency for p0'
    deps_p1 = G.dep(p1)
    assert len(deps_p1)==1 and (deps_p1!=p1), 'Corrupted adjacency for p1'
    deps_p2 = G.dep(p2)
    assert len(deps_p2)==1 and (deps_p2!=p2), 'Corrupted adjacency for p2'

  @unittest.skip('Temporarily removed device support, pending API review.')
  def test_devices(self):
    G = self.G
    for primop in G:
      assert primop.device is not None, 'Missing device on primop '+str(primop)
    for device in G.devices:
      assert device is not None, 'Missing device'
