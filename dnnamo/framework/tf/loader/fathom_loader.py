import importlib
import sys

from ....core.model import DnnamoModel
from ....core.loader import BaseLoader
from ..tf_model import session_run, SessionProfiler

FATHOM_MODELS = [
  'Seq2Seq',
  'MemNet',
  'Speech',
  'Autoenc',
  'Residual',
  'VGG',
  'AlexNet',
  'DeepQ',
]

class FathomModel(DnnamoModel):
  def __init__(self, ModelClass, ModelClassFwd):
    self._fathommodel_train = ModelClass()
    self._fathommodel_inf = ModelClassFwd()
    self._fathommodel_train.setup()
    self._fathommodel_inf.setup()

  def get_training_graph(self):
    return self._fathommodel_train.G

  def get_inference_graph(self):
    return self._fathommodel_inf.G

  def get_weights(self, keys=None):
    raise NotImplementedError, 'Fathom does not have a standard weights interface.'

  def profile_training(self, n_steps=1, *args, **kwargs):
    profiler = SessionProfiler()
    self._fathommodel_train.run(runstep=profiler.session_profile, n_steps=n_steps)
    return profiler.rmd

class TFFathomLoader(BaseLoader):
  def __init__(self, identifier, fathompath=None):
    '''Loader for the Fathom reference workloads.

    Fathom (https://github.com/rdadolf/fathom) contains reference implementations
    of several popular deep learning models. This loader allows them to be
    imported by name. The optional fathompath parameter can be used to point to
    the installation location of the Fathom package.'''

    if identifier not in FATHOM_MODELS:
      raise KeyError, 'No '+str(identifier)+' model found in Fathom. Valid models are: '+', '.join(FATHOM_MODELS)
    self.identifier = identifier
    self.fathompath = fathompath

  def load(self):
    # First, take care of loading the Fathom module.
    old_syspath = sys.path
    if self.fathompath is not None:
      sys.path.insert(0,self.fathompath)
    module = importlib.import_module('fathom')
    sys.path = old_syspath

    # Find and instantiate the corresponding Fathom module
    try:
      ModelClass = getattr(module, self.identifier)
    except KeyError:
      raise NameError, 'No '+str(self.identifier)+' class found in Fathom.'
    try:
      ModelClassFwd = getattr(module, self.identifier+'Fwd')
    except KeyError:
      raise NameError, 'No '+str(self.identifier)+'Fwd class found in Fathom.'

    m = FathomModel(ModelClass, ModelClassFwd)

    return m
