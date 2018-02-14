from functools import wraps
import pytest
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


@pytest.mark.parametrize('mode', ['training','inference'])
@pytest.mark.parametrize('model', ['test/test_models/simple_nnet'])
#@pytest.mark.xfail(raises=NotImplementedError)
class TestTFFrameworkDatatagAccessors(object):
  def test_datatag_graph_all_static_native(self, model, mode):
    frame = TFFramework(RunpyLoader, model)
    g = frame.get_graph(mode, 'static', 'native')
    assert len(g.ops)>0, 'No operations in graph.'

  def test_datatag_graph_all_static_primitive(self, model, mode):
    frame = TFFramework(RunpyLoader, model)
    g = frame.get_graph(mode, 'static', 'primitive')
    assert len(g.ops)>0, 'No operations in graph.'

  def test_datatag_graph_all_dynamic_native(self, model, mode):
    frame = TFFramework(RunpyLoader, model)
    g = frame.get_graph(mode, 'dynamic', 'native')
    assert len(g.ops)>0, 'No operations in graph.'

  def test_datatag_graph_all_dynamic_primitive(self, model, mode):
    frame = TFFramework(RunpyLoader, model)
    g = frame.get_graph(mode, 'dynamic', 'primitive')
    assert len(g.ops)>0, 'No operations in graph.'

  # FIXME: weights

  #def test_datatag_timing_all_dynamic_native(self, model, mode):
  #  frame = TFFramework(RunpyLoader, model)
  #  t = frame.get_timing(mode, 'native')
  #  assert len(t)>0, 'No timing information in profile.'

  #def test_datatag_timing_all_dynamic_primitive(self, model, mode):
  #  frame = TFFramework(RunpyLoader, model)
  #  t = frame.get_timing(mode, 'primitive')
  #  assert len(t)>0, 'No timing information in profile.'
