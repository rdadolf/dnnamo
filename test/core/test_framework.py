import unittest

from dnnamo.core.framework import Framework
from dnnamo.loader import RunpyLoader

class ExampleFramework(Framework):
  @property
  def translator(self):
    return None # We won't be using this, but translator is an abstract prop
    

class TestFramework(unittest.TestCase):
  def test_instantiation(self):
    frame = ExampleFramework()

  def test_collect_generics(self):
    frame = ExampleFramework()
    frame.load(RunpyLoader, 'test/test_models/simple_nnet.py')

    g = frame.get_graph(mode='training',scope='static',ops='native')
    assert g is not None, 'No training graph returned from get_graph'

    g = frame.get_graph(mode='inference',scope='static',ops='native')
    assert g is not None, 'No inference graph returned from get_graph'

    #g = frame.get_graph(mode='training',scope='static',ops='primitive')
    #assert g is not None, 'No abstract training graph returned from get_graph'

    #g = frame.get_graph(mode='inference',scope='static',ops='primitive')
    #assert g is not None, 'No abstract inference graph returned from get_graph'
