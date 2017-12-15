import timeit
import os

import tensorflow as tf

#from dnnamo.core.trace import Trace
from dnnamo.frameworks.tf.tf_trace import TFTracepoint, TFTrace

class _DefaultRunstep(object):
  def __init__(self): pass
  def __call__(self, session, fetches, *options, **kw_options):
    return session.run(fetches, *options, **kw_options)

class _InstrumentedRunstep(object):
  def __init__(self):
    self.native_model = None
    self.traces = []
    self.elapsed = None
    self.cpu_elapsed = None
    self.n_untraced_ops = 0

  def _parse_op_label(self,label):
    # This is apparently the appropriate way to extract the operation type
    # from a RunMetadata protobuf. (c.f.-tf/python/client/timeline.py)
    return label[label.index('=')+2:label.index('(')]

  def _runmetadata_to_trace(self, md):
    trace = TFTrace()
    # Structure of RunMetadata protobuf comes from two places in TF:
    #   tensorflow/core/protobuf/config.proto (just the top-level protobuf)
    #   tensorflow/core/framework/step_stats.proto (most everything else)

    for device in md.step_stats.dev_stats:
      for node in device.node_stats:

        optype = self._parse_op_label(node.timeline_label)
        tensor_dims = [[dim.size for dim in out.tensor_description.shape.dim] for out in node.output]
        trace.append(TFTracepoint(
          name = str(node.node_name),
          type = str(optype), #str(self.typemap[node.node_name]),
          device = str(device.device),
          dt = int(node.all_end_rel_micros),
          t0 = int(node.all_start_micros),
          tensor_dims = tensor_dims))

    return trace

  def __call__(self, session, fetches, *options, **kw_options):
    # FIXME: Really, we should be able to handle multiple requests for tracing.
    #   The tracing is done within TF anyways, so it's just a matter of copying
    #   the protobuf data to the model's location as well as our own.
    #   The issue is that the RunOptions proto has more options we'd need to
    #   preserve, and we'd need an in-place copier for the trace output proto.
    assert 'options' not in kw_options, 'Instrumentation collision: tracing already enabled in model run. Multiple tracers not yet implemented.'
    assert 'run_metadata' not in kw_options, 'Instrumentation collision: tracing already enabled in model run. Multiple tracers not yet implemented.'

    kw_options['options'] = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
    runmetadata = tf.RunMetadata()
    kw_options['run_metadata'] = runmetadata

    t0 = timeit.default_timer()
    cpu_t0 = os.times()
    retval = session.run(fetches, *options, **kw_options)
    t1 = timeit.default_timer()
    cpu_t1 = os.times()

    trace = self._runmetadata_to_trace(runmetadata)
    self.traces.append( trace )
    self.elapsed = t1-t0
    self.cpu_elapsed = (cpu_t1[0]-cpu_t0[0]) + (cpu_t1[1]-cpu_t0[1])
    # FIXME: remove these and add a real interface to this class to capture them
    print 'Wall elapsed:',self.elapsed
    print 'CPU elapsed:',self.cpu_elapsed

    trace.remove_generated_ops( session.graph )

    #for tp in trace:
    #  print tp

    return retval

