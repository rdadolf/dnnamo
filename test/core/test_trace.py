import unittest

import numpy as np

from dnnamo.core.trace import Tracepoint, Trace, average_traces

class TestTrace(unittest.TestCase):
  def test_basic_trace_functions(self):
    t = Trace()

    for i in xrange(0,100):
      t.append(Tracepoint(
        name='tp'+str(i),
        type='example_type',
        device='example_device',
        dt=i))

    assert len(t)==100, 'Incorrect length in trace construction.'
    assert t[13].dt==13, 'Lookup failure in trace using ordinal indexing.'
    assert t['tp'+str(13)].dt==13, 'Lookup failure in trace using name indexing.'
    s = 0
    for tp in t:
      s += tp.dt
    assert s==sum(xrange(0,100)), 'dt aggregation failed while iterating over trace.'

  def test_trace_averaging(self):
    traces = []
    mean_offsets = range(0,10)
    for mean_offset in mean_offsets:
      trace = Trace()
      for i in xrange(0,100):
        trace.append(Tracepoint(
          name='tp'+str(i),
          type='example_type',
          device='example_device',
          dt=i+mean_offset))
      traces.append(trace)
    t = average_traces(traces)

    computed_dt = 13+np.mean(mean_offsets)
    assert np.abs(t[13].dt - computed_dt)<0.0001, 'Incorrect average value for 13th tracepoint: '+str(t[13].dt)+' vs '+str(computed_dt)

    s = 0
    for tp in t:
      s += tp.dt
    computed_s = np.sum(np.arange(0,100)+np.mean(mean_offsets))
    assert s == computed_s, 'Incorrect dt sum for averaged trace: '+str(s) +' vs. '+str(computed_s)
