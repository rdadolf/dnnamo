from nnmodel.frameworks.tf import TFModel

class EmptyTFModel(TFModel):
  def model(self): return None
  def run(self, runstep=None, *args, **kwargs): return None
