import numpy as np

from ..framework import FRAMEWORKS
from .tool_utilities import BaselineTool, ToolRegistry

class DensityTool(BaselineTool):
  TOOL_NAME='density'
  TOOL_SUMMARY='Prints a list of the computational density (microseconds of compute per value consumed) of each op, sorted by total execution time'

  def _run(self):
    self.data = {} # model -> [(id,time,density), ...]
    for model in self.args['models']:
      frame = FRAMEWORKS[self.args['framework']]()
      frame.load(self.args['loader'], model, **self.args['loader_opts'])

      g = frame.get_graph(mode=self.args['mode'], scope='dynamic', ops='native')
      p = frame.get_timing(mode=self.args['mode'], ops='native')

      # FIXME: better aggregation
      timing = p.aggregate('last')

      dat = []
      for op in g.ops:
        id = op.id
        time = timing[op.id]
        # FIXME: really should be in bytes, not "values"
        values = int(sum([np.prod(g[t].shape) for t in g.tensors_to(op.id)]))
        dat.append( (id,time,values) )

      dat.sort(key=lambda (i,t,v): t, reverse=True)
      self.data[model] = dat

  def _output(self):
    for model,dat in self.data.items():
      print '---Model: '+str(model)+'---'
      for (id,time,values) in dat:
        s = '  '+str(time)+' us\t'+str(values)+' values\t'
        if values==0:
          s += '(no data)'
        else:
          s += str(np.round(float(time)/values, decimals=2))+'us/value'
        s += '\t'+str(id)
        print s

ToolRegistry.register(DensityTool.TOOL_NAME, DensityTool)
