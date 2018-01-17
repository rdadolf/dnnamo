import unittest

from dnnamo.core.model import DnnamoModel

class M1(DnnamoModel):
  def get_graph(self): return None # framework-specific return, can be anything
class M2(M1):
  def get_weights(self,keys=None): return dict() # any dict, even an empty one, is valid
class M3(M2):
  def set_weights(self, kv): pass
class M4(M3):
  def run_inference(self, n_steps=1, *args, **kwargs): pass
class M5(M4):
  def profile_inference(self, n_steps=1, *args, **kwargs): pass
class M6(M5):
  def run_training(self, n_steps=1, *args, **kwargs): pass
class M5(M4):
  def profile_training(self, n_steps=1, *args, **kwargs): pass
class M6(M5):
  def get_intermediates(self, *args, **kwargs): return dict()


class TestModels(unittest.TestCase):
  def test_instantiation(self):
    for m in [M1,M2,M3,M4,M5,M6]:
      _ = m()

