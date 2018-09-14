import importlib
import os.path
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
  # FIXME: FathomModel's currently have a bit of a mismatch with DnnamoModel.
  #   Fathom assumed the construction of two separate models, one for training
  #   and one for inference. Dnnamo assumes only one which can be run in either
  #   mode. As a result, modifications to one model will not impact the other,
  #   despite what Dnnamo would like. This could be hacked by manually patching
  #   weights to and from each model, but I have not done so.

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

  def profile_inference(self, n_steps=1, *args, **kwargs):
    profiler = SessionProfiler()
    self._fathommodel_inf.run(runstep=profiler.session_profile, n_steps=n_steps)
    return profiler.rmd

class TFFathomLoader(BaseLoader):
  def __init__(self, identifier, fathompath=None, modulename='fathom'):
    '''Loader for the Fathom reference workloads.

    Fathom (https://github.com/rdadolf/fathom) contains reference implementations
    of several popular deep learning models. This loader allows them to be
    imported by name. The optional fathompath parameter can be used to point to
    the installation location of the Fathom package.'''

    if identifier not in FATHOM_MODELS:
      raise KeyError, 'No '+str(identifier)+' model found in Fathom. Valid models are: '+', '.join(FATHOM_MODELS)
    self.modulename = modulename
    self.identifier = identifier
    self.fathompath = fathompath

  def load(self):
    # First, take care of loading the Fathom module.
    old_syspath = sys.path
    if self.fathompath is not None:
      sys.path.insert(0,self.fathompath)
    self.module = importlib.import_module(self.modulename)
    sys.path = old_syspath

    # Find and instantiate the corresponding Fathom module
    try:
      ModelClass = getattr(self.module, self.identifier)
    except KeyError:
      raise NameError, 'No '+str(self.identifier)+' class found in Fathom.'
    try:
      ModelClassFwd = getattr(self.module, self.identifier+'Fwd')
    except KeyError:
      raise NameError, 'No '+str(self.identifier)+'Fwd class found in Fathom.'

    m = FathomModel(ModelClass, ModelClassFwd)

    return m

class TFFathomLiteLoader(TFFathomLoader):
  def __init__(self, identifier, fathompath=None, modulename='fathomlite'):
    '''Loader for the Fathom-lite reference workloads.

    Fathom-lite (https://github.com/rdadolf/fathom-lite) is a lightweight
    alternative to the original Fathom reference workloads which uses smaller
    datasets in order to accurately capture performance behavior without the 
    cost of larger training datasets. This loader allows Fathom-lite models to be
    imported by name. The optional fathompath parameter can be used to point to
    the installation location of the Fathom package.'''

    TFFathomLoader.__init__(self, identifier, fathompath=fathompath, modulename=modulename)

  def load(self):
    m = TFFathomLoader.load(self)
    # Attempt to set the fathomlite data directory automatically
    assert self.module is not None, 'Module import failed.'
    fathomlite_directory = os.path.dirname(os.path.abspath(self.module.__file__))
    fathomlite_root = os.path.dirname(fathomlite_directory)
    fathomlite_data = os.path.join(fathomlite_root, 'data')
    self.module.Config.set('data_dir', fathomlite_data)
    return m
