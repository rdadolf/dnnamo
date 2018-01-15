import dnnamo
import dnnamo.frameworks
import dnnamo.devices
from dnnamo.loader import RunpyLoader
from dnnamo.core.trace import average_traces, Tracepoint, Trace
from .tool_utilities import BaselineTool, ToolRegistry

class MimicTool(BaselineTool):
  TOOL_NAME='_mimic'
  TOOL_SUMMARY='[DEV] Attempts to estimate the performance of a neural network by decomposing it into abstract pieces and then reassembling it.'

  def __init__(self):
    super(MimicTool,self).__init__()
    self.data = {}

  def add_subparser(self, argparser):
    super(MimicTool,self).add_subparser(argparser)
    self.subparser.add_argument('--detail','-d', default='profile', choices=['profile','framework','primop','regression'], help='The level of abstraction to which the neural net is decomposed before being reassembled.')
    self.subparser.add_argument('--full', default=False, action='store_true', help='Prints timing information for each individual component being estimated.')
    return self.subparser

  def _run(self, models):
    for model in models:
      frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]()
      frame.load(self.args['loader'], model, **self.args['loader_opts'])

      actions = { 'profile': self._mimic_profile,
                  'framework': self._mimic_framework,
                  'primop': self._mimic_primop,
                  'regression': self._mimic_regression }
      
      _,true_time = frame.run(n_steps=1)
      mimic_time, components = actions[self.args['detail']](frame)
      self.data[model] = (true_time, mimic_time, components)
      return True

  def _output(self):
    super(MimicTool,self)._output()

    for model,(true_time, mimic_time, components) in self.data.items():
      precision = 2 # max decimal places
      print 'True vs. Mimic time: '+str(round(true_time,precision))+'us '+str(round(mimic_time,precision))+'us ('+str(round(mimic_time*100/true_time,precision))+'%)'
      if self.args['full']:
        print 'Component timing information:'
        if components is None:
          print '  No components found.' # mostly for empty cachefiles
          return
        for name,timing in components.items():
          print '  '+str(name)+':',timing

  def _mimic_profile(self, frame):
    timing_info = frame.get_timing()
    t_sum = sum([usecs for op,usecs in timing_info.items()])
    return (t_sum, timing_info)

  def _mimic_framework(self, frame):
    return (0,None)

  def _mimic_primop(self, frame):
    return (0,None)

  def _mimic_regression(self, frame):
    return (0,None)


ToolRegistry.register(MimicTool)
