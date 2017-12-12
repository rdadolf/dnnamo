from synth import DGraphTestCase

class TestDGraph(DGraphTestCase):
  def test_adjacency_constraints(self):
    DG = self.DG
    assert len(DG)==3, 'Added three primops, graph is not three nodes'
    (p0,p1,p2) = DG.primops.values()
    # Check simple edge invariants
    deps_p0 = DG.dep(p0)
    assert len(deps_p0)==1 and (deps_p0!=p0), 'Corrupted adjacency for p0'
    deps_p1 = DG.dep(p1)
    assert len(deps_p1)==1 and (deps_p1!=p1), 'Corrupted adjacency for p1'
    deps_p2 = DG.dep(p2)
    assert len(deps_p2)==1 and (deps_p2!=p2), 'Corrupted adjacency for p2'

  def test_devices(self):
    DG = self.DG
    for primop in DG:
      assert primop.device is not None, 'Missing device on primop '+str(primop)
    for device in DG.devices:
      assert device is not None, 'Missing device'
