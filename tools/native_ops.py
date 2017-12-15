import dnnamo
import dnnamo.frameworks
from tool_utilities import BaselineTool

class Tool(BaselineTool):
  TOOL_NAME='native_ops'
  TOOL_SUMMARY='Prints a list of the framework-specific operations in a model.'

  def __init__(self):
    super(Tool,self).__init__()
    self.data = []

  def _run(self, modelfiles):
    for modelfile in modelfiles:
      Frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]
      frame = Frame()
      frame.load(modelfile)
      m = frame.native_model()
      if self.args['framework']=='tf':
        ops = [(op.name, op.type, op.device) for op in m.get_operations()]
        self.data.append(ops)
      else:
        assert False, 'Unknown framework "'+str(self.args['framework'])+'"'

  def _output(self):
    for ops in self.data:
      for op in ops:
        print ','.join(map(str,op))

