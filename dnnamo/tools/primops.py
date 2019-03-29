from ..framework import FRAMEWORKS
from .tool_utilities import BaselineTool, ToolRegistry

class PrimopsTool(BaselineTool):
  TOOL_NAME='primops'
  TOOL_SUMMARY='Prints a list of the primitive operations in a model.'
  CACHE_FORMAT = [
    'primops', # [(id,type), ...]
  ]

  def __init__(self):
    super(PrimopsTool,self).__init__()

  def add_subparser(self, argparser):
    super(PrimopsTool,self).add_subparser(argparser)
    self.subparser.add_argument('--run',action='store_true',default=False,help='Collect data from an execution trace')
    self.subparser.add_argument('--timing',action='store_true',default=False,help='Collect and sort primops by execution time (implies --run)')
    self.subparser.add_argument('--undef',action='store_true',default=False,help='Display undefined Primops')
    return self.subparser

  def _run(self):
    if self.args['timing']:
      self.args['run'] = True
    model = self.args['model']
    frame = FRAMEWORKS[self.args['framework']]()
    print self.args['loader_opts']
    frame.load(self.args['loader'], model, **self.args['loader_opts'])
    if self.args['run']:
      g = frame.get_graph(mode=self.args['mode'],scope='dynamic',ops='primitive')
    else:
      g = frame.get_graph(mode=self.args['mode'],scope='static',ops='primitive')

    if self.args['timing']:
      p = frame.get_timing(mode=self.args['mode'],ops='primitive').aggregate('last')
      self.data['primops'] = [(op.id, op.type, op.root.type, p[op.id]) for op in g.ops if op.type!='undef' or self.args['undef']]
      self.data['primops'].sort(key=lambda (_0,_1,_2,t): t, reverse=True)
    else:
      self.data['primops'] = [(op.id, op.type, op.root.type, None) for op in g.ops if op.type!='undef' or self.args['undef']]

  def _output(self):
    for op_id,op_type,root_type,timing in self.data['primops']:
      s = '  '
      if timing is not None:
        s += str(timing)+'us\t'
      s += '\t'+str(op_type)+' ('+str(root_type)+')\t'+str(op_id)
      print s

ToolRegistry.register(PrimopsTool.TOOL_NAME, PrimopsTool)
