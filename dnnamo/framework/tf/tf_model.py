import tensorflow as tf

# Support functions for writing TensorFlow wrappers
#
# These methods supply some of the common functionality that users will need
# when creating a Dnnamo model wrapper. These are not a replacement for a model
# wrapper.

def session_run(session, fetches, feed_dict=None):
  '''Replacement for tf.Session.run(...) which returns profiling data.

  The first argument must be a tf.Session().
  This is basically a pass-through wrapper function. It exists mainly to provide
  an argument-identical dual to session_profile().
  Note that this function does not accept the options or run_metadata arguments
  to tf.Session.run().'''
  return session.run(fetches, feed_dict, options=None, run_metadata=None)


def session_profile(session, fetches, feed_dict=None):
  # NOTE: If you use this function, you must manually aggregate multiple
  # steps. I.e., the model interface method "profile_training" expects a
  # list of rmd structures, so you must manually combine them in this
  # function's caller.

  '''Partial replacement for tf.Session.run(...) which returns profiling data.

  The first argument must be a tf.Session().
  Note that this function does not accept the options or run_metadata arguments
  to tf.Session.run(). Additionally, this function returns runmetadata, not
  the normal output of Session.run(), so if your model needs it, use the
  stateful helper class SessionProfiler.'''

  opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
  rmd = tf.RunMetadata()
  _ = session.run(fetches, feed_dict, options=opts, run_metadata=rmd)
  return rmd

###

class SessionProfiler(object):
  '''A stateful wrapper for tf.Session.run. When used, profiling data is recorded in the .rmd attribute.'''

  def __init__(self):
    self.rmd = []
  
  def session_profile(self, session, fetches, feed_dict=None):
    '''Replacement for tf.Session.run(...) which returns profiling data.

    The first argument must be a tf.Session().
    Note that this function does not accept the options or run_metadata arguments
    to tf.Session.run().'''
    opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
    self.rmd.append(tf.RunMetadata())
    out = session.run(fetches, feed_dict, options=opts, run_metadata=self.rmd[-1])
    return out
