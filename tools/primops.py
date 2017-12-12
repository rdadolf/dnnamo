import nnmodel
import nnmodel.frameworks
from tool_utilities import BaselineTool

class Tool(BaselineTool):
  TOOL_NAME='primops'
  TOOL_SUMMARY='Prints a list of the primitive operations in a model.'

  def __init__(self):
    super(Tool,self).__init__()
    self.data = []

  def add_subparser(self, argparser):
    super(Tool,self).add_subparser(argparser)
    self.subparser.add_argument('--undef',action='store_true',default=False,help='Display undefined Primops.')
    return self.subparser

  def _run(self, modelfiles):
    for modelfile in modelfiles:
      frame = nnmodel.frameworks.FRAMEWORKS[self.args['framework']]()
      frame.load(modelfile)
      ops = [(primop.id,primop.optype,primop.device) for primop in frame.graph() if primop.optype!='undef' or self.args['undef']]
      self.data.append(ops)

  def _output(self):
    for ops in self.data:
      for op in ops:
        print ','.join(map(str,op))
