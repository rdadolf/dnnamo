class DnnamoModel(object):
  @property
  def is_dnnamo_model(self):
    '''A property which can be used to identify unknown objects as Dnnamo models.'''
    return True

  def get_inference_graph(self):
    '''Returns a computational graph which includes the inference path of model.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def get_training_graph(self):
    '''Returns a computational graph which includes the training path of model.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def get_weights(self, keys=None):
    '''Returns a dictionary of the current weight values in the model.

    In the return value, keys are framework-specific string identifiers, and
    values are multi-dimensional numpy arrays.

    The options keys parameter allows only a subset of weights to be selected.
    This parameter can be any iterable, with key-like elements.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def set_weights(self, kv):
    '''Sets the specified weight values in the model.

    The kv parameter is a dictionary where keys are framework-specific string
    identifiers, and values are multi-dimensional numpy arrays of the correct
    size. Weights in the model but omitted in the kv dictionary will retain
    their previous values.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def run_inference(self, n_steps=1, *args, **kwargs):
    '''Produce model estimates for a fixed number of inputs.

    Returns a list of output values of length n_steps.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def profile_inference(self, n_steps=1, *args, **kwargs):
    '''Collect timing information for a fixed number of inputs.

    Returns framework-specific profile data.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def run_training(self, n_steps=1, *args, **kwargs):
    '''Trains the model for a fixed number of training steps.

    Returns a list of loss values of length n_steps.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def profile_training(self, n_steps=1, *args, **kwargs):
    '''Collects timing information for a fixed number of training steps.

    Returns framework-specific profile data.'''
    raise NotImplementedError, 'This model does not implement this method.'

  def get_intermediates(self, *args, **kwargs):
    '''Produce a dictionary of all intermediate values for a single input.'''
    raise NotImplementedError, 'This model does not implement this method.'
    # FIXME: Return an "IValues" object or something instead?
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
