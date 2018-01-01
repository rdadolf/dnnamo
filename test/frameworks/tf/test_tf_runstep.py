import unittest

from dnnamo.frameworks.tf import TFFramework

class TestTFRunstep(unittest.TestCase):
  def test_existence_of_runsteps(self):
    frame = TFFramework()
    default_runner = frame.DefaultRunstep()
    instrumented_runner = frame.DefaultRunstep()
    assert hasattr(default_runner,'__call__'), 'TFFramework.DefaultRunner cannot be called.'
    assert hasattr(instrumented_runner, '__call__'), 'TFFramework.InstrumentedRunner cannot be called.'

  def test_run_with_default_runstep(self):
    frame = TFFramework()
    frame.load('test/test_models/simple_nnet.py')
    frame.run(n_steps=1)

  def test_run_native_trace(self):
    frame = TFFramework()
    frame.load('test/test_models/simple_nnet.py')
    traces = frame.run_native_trace(n_steps=1)
    assert isinstance(traces,list), 'TFFramework.run_native_trace returned an invalid result.'
    assert len(traces)>0, 'TFFramework.run_native_trace did not capture any traces.'
    assert len(traces)<2, 'TFFramework.run_native_trace returned corrupted data.'

  @unittest.SkipTest
  def test_run_trace(self):
    frame = TFFramework()
    frame.load('test/test_models/simple_nnet.py')
    traces = frame.run_trace(n_steps=1)
    assert isinstance(traces,list), 'TFFramework.run_trace returned an invalid result.'
    assert len(traces)>0, 'TFFramework.run_trace did not capture any traces.'
    assert len(traces)<2, 'TFFramework.run_trace returned corrupted data.'
