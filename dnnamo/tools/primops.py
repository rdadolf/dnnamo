from ..framework import FRAMEWORKS
from ..loader import RunpyLoader
from .tool_utilities import BaselineTool, ToolRegistry

class PrimopsTool(BaselineTool):
  TOOL_NAME='primops'
  TOOL_SUMMARY='Prints a list of the primitive operations in a model.'

  def __init__(self):
    super(PrimopsTool,self).__init__()
    self.data = []

  def add_subparser(self, argparser):
    super(PrimopsTool,self).add_subparser(argparser)
    self.subparser.add_argument('--undef',action='store_true',default=False,help='Display undefined Primops.')
    return self.subparser

  def _run(self, models):
    for model in models:
      frame = FRAMEWORKS[self.args['framework']]()
      print self.args['loader_opts']
      frame.load(self.args['loader'], model, **self.args['loader_opts'])
      # FIXME: model mode should be selectable from the CLI. Hardcoding it to
      #   'training' is the wrong thing.
      ops = [(primop.id,primop.optype) for primop in frame.get_graph(mode='training',scope='static',ops='primitive') if primop.optype!='undef' or self.args['undef']]
      self.data.append(ops)

  def _output(self):
    for ops in self.data:
      for op in ops:
        print ','.join(map(str,op))

ToolRegistry.register(PrimopsTool)
