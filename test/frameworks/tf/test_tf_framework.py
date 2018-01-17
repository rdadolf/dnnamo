import unittest
from dnnamo.core.model import DnnamoModel
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
        assert isinstance(frame.model, DnnamoModel), 'Model isnt actually a Dnnamo model'

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

