from dnnamo.core.model import ImmutableModel, StaticModel, DynamicModel

class EmptyImmutableModel(ImmutableModel):
  def get_graph(self): return None
  def get_weights(self): return dict()
class EmptyStaticModel(EmptyImmutableModel):
  def set_weights(self, kv): pass
class EmptyDynamicModel(EmptyStaticModel):
  def run_train(self, runstep=None, n_steps=1, *args, **kwargs): pass
  def run_inference(self, runstep=None, n_steps=1, *args, **kwargs): pass
  def get_activations(self, runstep=None, *args, **kwargs): return dict()
