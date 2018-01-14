import unittest
from dnnamo.core.model import BaseModel
from dnnamo.frameworks.tf import TFFramework
from dnnamo.loader import RunpyLoader

#FIXME
#from synth import TFTestCase

class TestTFFramework(unittest.TestCase):
  def test_load(self):
    loaders = [
      RunpyLoader,
    ]
    identifiers = [
      'test.test_models.empty_models',
      'test/test_models/empty_models.py',
      'test/../test/test_models/empty_models.py', # FIXME
    ]
    for loader in loaders:
      for identifier in identifiers:
        frame = TFFramework()
        print 'Loading:',identifier
        frame.load(loader, identifier)
        assert frame.model.is_dnnamo_model, 'Model isnt actually a Dnnamo model'
        assert isinstance(frame.model, BaseModel), 'Model isnt actually a Dnnamo model'

  def test_failed_load(self):
    with self.assertRaises(ImportError):
      frame = TFFramework()
      frame.load(RunpyLoader, 'nonexistent_module')

  @unittest.SkipTest
  # FIXME: NYI
  def test_model_instance_in_constructor(self):
    # model = ...
    #frame = TFFramework(model)
    pass

  @unittest.SkipTest
  def test_basic_graph_extraction(self):
    g = self.synth_ff_network()
    t = TFFramework(g)
    absgraph = t.absgraph()

    nodes = len(absgraph)
    print 'nodes:',nodes
    assert nodes>0, 'Empty dependence graph'

    edges = 0
    for p in absgraph:
      edges += len(absgraph.dep(p))
    print 'edges:',edges
    assert edges>=nodes-1, 'Disconnected graph'
    # FIXME: more testing here

  def test_transitive_closure(self):
    frame = TFFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet')
    m = frame.model
    closed_ops = frame._transitive_closure([m.loss,m.train])
    all_ops = m.get_graph().get_operations()
    trace = frame.run_native_trace(n_steps=3)[1]

    closed_names = set([op.name for op in closed_ops])
    all_names = set([op.name for op in all_ops])
    traced_names = set([tracepoint.name for tracepoint in trace])

    assert len(closed_names - all_names)==0, 'Unknown operators found in transitive closure'
    assert len(traced_names - all_names)==0, 'Unknown operators found in trace'

    traced_not_closed = traced_names - closed_names
    assert len(traced_not_closed)==0, 'Found names in native trace that were not in the transitive closure: '+str(traced_not_closed)
    # FIXME: more testing here

  @unittest.SkipTest # FIXME: this test is broken (it uses obsolete interfaces)
  def test_device_mapping(self):
    g = self.synth_ff_network()
    t = tframe.TFFramework(g)
    absgraph = t.absgraph()

    for primop in absgraph:
      assert primop.device is not None

    assert len(set(absgraph.devices))==1, 'More than one device found'

