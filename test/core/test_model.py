import unittest

from nnmodel.core.model import Model

class BADSubModel(Model): pass

class SubModel(Model):
  def model(self): pass
  def run(self, runstep=None, *args, **kwargs): pass

class TestModel(unittest.TestCase):
  def test_model_class_abstractness(self):
    with self.assertRaises(TypeError):
      #pylint: disable=unused-variable
      m = Model()

  def test_model_subclass_abstractness(self):
    with self.assertRaises(TypeError):
      #pylint: disable=unused-variable
      m = BADSubModel()

  def test_model_override_functions(self):
    # If new abstractmethods are added, this should pick them up
    #pylint: disable=unused-variable
    m = SubModel()
