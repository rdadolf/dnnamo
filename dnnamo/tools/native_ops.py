import dnnamo
import dnnamo.frameworks
from dnnamo.loader import RunpyLoader
from .tool_utilities import BaselineTool, ToolRegistry

class NativeOpsTool(BaselineTool):
  TOOL_NAME='native_ops'
  TOOL_SUMMARY='Prints a list of the framework-specific operations in a model.'

  def __init__(self):
    super(NativeOpsTool,self).__init__()
    self.data = []

  def _run(self, models):
    for model in models:
      frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]()
      frame.load(self.args['loader'], model, **self.args['loader_opts'])

      if self.args['framework']=='tf':
        ops = [(op.name, op.type, op.device) for op in frame.graph.get_operations()]
        self.data.append(ops)
      else:
        assert False, 'Unknown framework "'+str(self.args['framework'])+'"'

  def _output(self):
    for ops in self.data:
      for op in ops:
        print ','.join(map(str,op))

ToolRegistry.register(NativeOpsTool)
