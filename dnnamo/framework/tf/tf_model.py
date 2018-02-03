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
  '''Replacement for tf.Session.run(...) which returns profiling data.

  The first argument must be a tf.Session().
  Note that this function does not accept the options or run_metadata arguments
  to tf.Session.run().'''
  opts = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE,output_partition_graphs=True)
  rmd = tf.RunMetadata()
  session.run(fetches, feed_dict, options=opts, run_metadata=rmd)
  return rmd
