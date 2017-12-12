import tensorflow as tf

from ...core.model import Model

class TFModel(Model):
  # pylint: disable=abstract-method
  # (the user inheriting from a Model needs to implement run() and model() )

  def __init__(self, device=None, init_options=None):
    '''An error-checking constructor.

    This is largely intended to be called via super() in child classes.'''
    super(TFModel, self).__init__(device, init_options)
    if device is not None:
      try: # Catch invalid devices early, before we waste time building things.
        _ = tf.DeviceSpec.from_string(device)
      except AttributeError:
        pass
      # FIXME: This except block exists because at the time of writing, we were
      # using TF v0.8.0, in which DeviceSpec's do not exist. As of 5/12/16, the
      # master branch of TF has this, which allows us to perform the check.
      # However, it didn't make sense to bump the TF version number just for
      # this. So the check is skipped if it doesn't exist. Once we bump TF, this
      # check should start to work, and this except block can (and should) be
      # removed.
      self.device = device

  def setup(self, setup_options=None):
    '''An error-checking setup function.

    This is largely intended to be called via super() in child classes.'''

    # Ignore all unknown options
    sess_options = {}
    if setup_options is not None:
      config_proto_keywords = [f.name for f in tf.ConfigProto.DESCRIPTOR.fields]
      for k in setup_options:
        if k in config_proto_keywords:
          sess_options[k] = setup_options[k]
        else:
          print 'Warning: invalid setup option: '+str(k)+' = '+str(setup_options[k])
    return sess_options

