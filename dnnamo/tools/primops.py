from ..framework import FRAMEWORKS
from ..loader import RunpyLoader
from .tool_utilities import BaselineTool, ToolRegistry

class PrimopsTool(BaselineTool):
  TOOL_NAME='primops'
  TOOL_SUMMARY='Prints a list of the primitive operations in a model.'

  def __init__(self):
    super(PrimopsTool,self).__init__()
    self.data = {} # model_identifier -> [(id,type), ...]

  def add_subparser(self, argparser):
    super(PrimopsTool,self).add_subparser(argparser)
    self.subparser.add_argument('--run',action='store_true',default=False,help='Collect data from an execution trace')
    self.subparser.add_argument('--timing',action='store_true',default=False,help='Collect and sort primops by execution time (implies --run)')
    self.subparser.add_argument('--undef',action='store_true',default=False,help='Display undefined Primops')
    return self.subparser

  def _run(self, models):
    if self.args['timing']:
      self.args['run'] = True
    for model in models:
      frame = FRAMEWORKS[self.args['framework']]()
      print self.args['loader_opts']
      frame.load(self.args['loader'], model, **self.args['loader_opts'])
      # FIXME: model mode should be selectable from the CLI. Hardcoding it to
      #   'training' is the wrong thing.
      if self.args['run']:
        g = frame.get_graph(mode='training',scope='dynamic',ops='primitive')
      else:
        g = frame.get_graph(mode='training',scope='static',ops='primitive')

      if self.args['timing']:
        p = frame.get_timing(mode='training',ops='primitive').aggregate('last')
        self.data[model] = [(op.id, op.type, op.root.type, p[op.id]) for op in g.ops if op.type!='undef' or self.args['undef']]
        self.data[model].sort(key=lambda (_0,_1,_2,t): t, reverse=True)
      else:
        self.data[model] = [(op.id, op.type, op.root.type, None) for op in g.ops if op.type!='undef' or self.args['undef']]

  def _output(self):
    for model,ops in self.data.items():
      print '---Model: '+str(model)+'---'
      for op_id,op_type,root_type,timing in ops:
        s = '  '
        if timing is not None:
          s += str(timing)+'us\t'
        s += '\t'+str(op_type)+' ('+str(root_type)+')\t'+str(op_id)
        print s

ToolRegistry.register(PrimopsTool)
