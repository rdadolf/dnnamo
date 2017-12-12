from nnmodel.frameworks.tf import TFModel

class EmptyTFModel0(TFModel):
  def model(self): return None
  def run(self, runstep=None, *args, **kwargs): return None

class EmptyTFModel1(TFModel):
  def model(self): return None
  def run(self, runstep=None, *args, **kwargs): return None
