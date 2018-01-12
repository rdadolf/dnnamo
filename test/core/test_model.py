import unittest

from dnnamo.core.model import BaseModel, ImmutableModel, StaticModel, DynamicModel

class BADImmutableModel(ImmutableModel): pass
class GOODImmutableModel(ImmutableModel):
  def get_graph(self): return None # framework-specific return can be anything
  def get_weights(self,keys=None): return dict() # any dict, even an empty one, is valid

class BADStaticModel(StaticModel): pass
class GOODStaticModel(StaticModel):
  def get_graph(self): return None # framework-specific return can be anything
  def get_weights(self,keys=None): return dict() # any dict, even an empty one, is valid
  def set_weights(self, kv): pass

class BADDynamicModel(DynamicModel): pass
class GOODDynamicModel(DynamicModel):
  def get_graph(self): return None # framework-specific return can be anything
  def get_weights(self,keys=None): return dict() # any dict, even an empty one, is valid
  def set_weights(self, kv): pass
  def run_train(self, runstep=None, n_steps=1, *args, **kwargs): pass
  def run_inference(self, runstep=None, n_steps=1, *args, **kwargs): pass
  def get_activations(self, runstep=None, *args, **kwargs): return dict()


class TestModels(unittest.TestCase):
  def test_model_class_abstractness(self):
    with self.assertRaises(TypeError):
      _ = BaseModel()
    with self.assertRaises(TypeError):
      _ = BADImmutableModel()
    with self.assertRaises(TypeError):
      _ = BADStaticModel()
    with self.assertRaises(TypeError):
      _ = BADDynamicModel()

  def test_model_subclass_instantiation(self):
    # If new abstract methods are added to base classes (e.g., BaseModel), this
    # test should flag it.
    _ = GOODImmutableModel()
    _ = GOODStaticModel()
    _ = GOODDynamicModel()

