from dnnamo.core.model import DnnamoModel
from dnnamo.core.profile import Profile

class EmptyModel(DnnamoModel):
  def get_training_graph(self): return None
  def get_inference_graph(self): return None
  def get_weights(self, keys=None): return dict()
  def set_weights(self, kv): pass
  def run_inference(self, n_steps=1, *args, **kwargs): return [None]
  def profile_inference(self, n_steps=1, *args, **kwargs): return Profile()
  def get_intermediates(self, *args, **kwargs): return None
  def run_training(self, n_steps=1, *args, **kwargs): return [None]
  def profile_training(self, n_steps=1, *args, **kwargs): return Profile()

def __dnnamo_loader__():
  return EmptyModel()
