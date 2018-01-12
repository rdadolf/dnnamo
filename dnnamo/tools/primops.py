import dnnamo
import dnnamo.frameworks
from dnnamo.loader import RunpyLoader
from .tool_utilities import BaselineTool, path_to_loader_pair, ToolRegistry

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

  def _run(self, modelfiles):
    for modelfile in modelfiles:
      frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]()
      (modname, pypath) = path_to_loader_pair(modelfile)
      frame.load(RunpyLoader, modname, pypath=pypath)
      ops = [(primop.id,primop.optype,primop.device) for primop in frame.absgraph if primop.optype!='undef' or self.args['undef']]
      self.data.append(ops)

  def _output(self):
    for ops in self.data:
      for op in ops:
        print ','.join(map(str,op))

ToolRegistry.register(PrimopsTool)
