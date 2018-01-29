import unittest
from dnnamo.core.model import DnnamoModel
from dnnamo.framework.tf import TFFramework
from dnnamo.loader import RunpyLoader


class TestTFFramework(unittest.TestCase):
  def test_load(self):
    loaders = [
      RunpyLoader,
    ]
    identifiers = [
      'test.test_models.empty_model',
      'test/test_models/empty_model.py',
      'test/../test/test_models/empty_model.py',
      'test.test_models.simple_nnet',
      'test/test_models/simple_nnet.py',
      'test/../test/test_models/simple_nnet.py',
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

  def test_graph_datatag_accessors(self):
    frame = TFFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')
    assert frame.get_graph(mode='training') is not None, 'No training graph returned.'
    assert frame.get_graph(mode='inference') is not None, 'No inference graph returned.'

  def test_abstract_datatag_accessors(self):
    frame = TFFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')
    assert frame.get_graph(mode='training',scope='static',ops='primitive') is not None, 'No abstract graph returned.'
    assert frame.get_graph(mode='inference',scope='static',ops='primitive') is not None, 'No abstract graph returned.'

  # FIXME
  @unittest.skip('The test models dont know how to export weights yet.')
  def test_weight_datatag_accessors(self):
    frame = TFFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')
    assert frame.get_weights(mode='training') is not None, 'No weights returned.'
    assert frame.get_weights(mode='inference') is not None, 'No weights returned.'

  def test_get_timing(self):
    frame = TFFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')
    assert frame.get_timing(mode='training') is not None, 'No timing returned.'
    assert frame.get_timing(mode='inference') is not None, 'No timing returned.'
