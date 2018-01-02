from abc import ABCMeta, abstractmethod, abstractproperty

#FIXME: Obsolete, remove when new classes are tested.
#
#class Model(object):
#  __metaclass__ = ABCMeta
#
#  def __init__(self, device=None, init_options=None):
#    '''(optional)'''
#    pass
#
#  @abstractmethod
#  def model(self):
#    '''Return a reference to the native representation of the model.
#
#    The exact representation may differ from framework to framework, but it should
#    reflect a complete description of the model being run and measured.
#    This function is NOT guaranteed to be called before running the model.'''
#
#  def setup(self, setup_options=None):
#    '''(optional) Prepare the model for running.
#
#    This should always be called before running a model, but may not need to be
#    overridden for a given model. Input download or pre-processing is approrpriate
#    here.'''
#
#  @abstractmethod
#  def run(self, runstep=None, n_steps=1, *args, **kwargs):
#    '''Run the model.
#
#    This may train the model, perform inference, or do something else. It depends
#    entirely on the model provided. Regardless, it should execute the thing that
#    is being measured.'''
#
#  def teardown(self):
#    '''(optional) Clean up after a model run.
#
#    This should always be called after running a model, but may not need to be
#    overridden for a given model. Cache or temp file removal is appropriate here.'''


class BaseModel(object):
  __metaclass__ = ABCMeta
  @abstractproperty
  def is_dnnamo_model(self):
    '''The existence of is_dnnamo_model in the base class prevents instantiation.

    All other Dnnamo models have a concrete is_dnnamo_model which returns True.'''

class ImmutableModel(BaseModel):
  __metaclass__ = ABCMeta

  @property
  def is_dnnamo_model(self):
    '''A property which can be used to identify unknown objects as Dnnamo models.'''
    return True

  @abstractmethod
  def get_graph(self):
    '''Returns a framework object with the computational graph of the model.'''
  @abstractmethod
  def get_weights(self, keys=None):
    '''Returns a dictionary of the current weight values in the model.

    In the return value, keys are framework-specific string identifiers, and
    values are multi-dimensional numpy arrays.

    The options keys parameter allows only a subset of weights to be selected.
    This parameter can be any iterable, with key-like elements.'''

class StaticModel(ImmutableModel):
  __metaclass__ = ABCMeta

  @abstractmethod
  def set_weights(self, kv):
    '''Sets the specified weight values in the model.

    The kv parameter is a dictionary where keys are framework-specific string
    identifiers, and values are multi-dimensional numpy arrays of the correct
    size. Weights in the model but omitted in the kv dictionary will retain
    their previous values.'''

class DynamicModel(StaticModel):
  __metaclass__ = ABCMeta

  @abstractmethod
  def run_train(self, runstep=None, n_steps=1, *args, **kwargs):
    '''Trains the model for a fixed number of minibatch steps.

    Returns a list of loss values of length n_steps.'''
    # FIXME: Definition may change, and needs parameter documentation.

  @abstractmethod
  def run_inference(self, runstep=None, n_steps=1, *args, **kwargs):
    '''Produce model estimates for a fixed number of inputs.

    Returns a list of output values of length n_steps.'''
    # FIXME: Definition may change, and needs parameter documentation.

  @abstractmethod
  def get_activations(self, runstep=None, *args, **kwargs):
    '''Produce a dictionary of all intermediate values for a single input.'''
    # FIXME: Definition may change, and needs parameter documentation.

    # FIXME: The definition of "activation" is a bit tricky. If it's "any
    #   intermediate tensor in the computational graph", there are way too many
    #   to sift through. Defining them as the "output of a layer" gives us
    #   problems with the definition of a "layer"---some models don't really have
    #   them, and what do we do with stateful neurons? Just their output or their
    #   state, too?
    #   In the end, I think we're going to need some sort of tagging system.
    #   Since I don't know what that's going to look like right now, I'm leaving
    #   this problem for later. I know this function is going to be very useful
    #   for some people, but I need to put some thought into how to approach it.
