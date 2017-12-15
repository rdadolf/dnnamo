from abc import ABCMeta, abstractmethod

class Model(object):
  __metaclass__ = ABCMeta

  def __init__(self, device=None, init_options=None):
    '''(optional)'''
    pass

  @abstractmethod
  def model(self):
    '''Return a reference to the native representation of the model.

    The exact representation may differ from framework to framework, but it should
    reflect a complete description of the model being run and measured.
    This function is NOT guaranteed to be called before running the model.'''

  def setup(self, setup_options=None):
    '''(optional) Prepare the model for running.

    This should always be called before running a model, but may not need to be
    overridden for a given model. Input download or pre-processing is approrpriate
    here.'''

  @abstractmethod
  def run(self, runstep=None, n_steps=1, *args, **kwargs):
    '''Run the model.

    This may train the model, perform inference, or do something else. It depends
    entirely on the model provided. Regardless, it should execute the thing that
    is being measured.'''

  def teardown(self):
    '''(optional) Clean up after a model run.

    This should always be called after running a model, but may not need to be
    overridden for a given model. Cache or temp file removal is appropriate here.'''


