import unittest
from dnnamo.frameworks.tf import TFFramework

#FIXME
#from synth import TFTestCase

class TestTFFramework(unittest.TestCase):
  def test_load(self):
    names = [
      ('test/test_models/empty_models.py',None),
      ('test//../test/test_models/empty_models.py',None),
      ('test/test_models/empty_models.py','EmptyTFModel'),
      ('test//../test/test_models/empty_models.py','EmptyTFModel'),
      ('test/test_models/empty_models.py:EmptyTFModel',None),
      ('test//../test/test_models/empty_models.py:EmptyTFModel',None),
      ('test/test_models/empty_models.py:EmptyTFModel','EmptyTFModel'),
      ('test//../test/test_models/empty_models.py:EmptyTFModel','EmptyTFModel'),
      ('test/test_models/multiple_tf_models.py','EmptyTFModel1'),
      ('test//../test/test_models/multiple_tf_models.py','EmptyTFModel1'),
      ('test/test_models/multiple_tf_models.py:EmptyTFModel1',None),
      ('test//../test/test_models/multiple_tf_models.py:EmptyTFModel1',None),
      ('test/test_models/multiple_tf_models.py:EmptyTFModel1','EmptyTFModel1'),
      ('test//../test/test_models/multiple_tf_models.py:EmptyTFModel1','EmptyTFModel1'),
    ]
    for filename,modelname in names:
      frame = TFFramework()
      print 'Loading:',filename,modelname
      frame.load(filename,modelname)
      assert isinstance(frame.model(), TFModel), 'Model isnt actualy a TF model'
  def test_failed_load(self):
    with self.assertRaises(IOError):
      frame = TFFramework()
      frame.load('/nonexistent-path',None)
    with self.assertRaises(ImportError):
      frame = TFFramework()
      frame.load('/','nonexistent-model')

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
    dg = t.graph()

    nodes = len(dg)
    print 'nodes:',nodes
    assert nodes>0, 'Empty dependence graph'

    edges = 0
    for p in dg:
      edges += len(dg.dep(p))
    print 'edges:',edges
    assert edges>=nodes-1, 'Disconnected graph'
    # FIXME: more testing here

  def test_transitive_closure(self):
    frame = TFFramework()
    frame.load('test/test_models/simple_nnet.py')
    m = frame.model()
    closed_ops = frame._transitive_closure([m.loss,m.train])
    all_ops = m.model().get_operations()
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
    dg = t.graph()

    for primop in dg:
      assert primop.device is not None

    assert len(set(dg.devices))==1, 'More than one device found'

