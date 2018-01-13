import tensorflow as tf

from ..frameworks import FRAMEWORKS
from ..loader import RunpyLoader
from .tool_utilities import BaselineTool, path_to_loader_pair, ToolRegistry

def dimstr(dim):
  if dim.value is None:
    return '?'
  else:
    return str(dim.value)
def format_dims(tf_tensor):
  try:
    return '['+'x'.join([dimstr(d) for d in tf_tensor.shape])+']'
  except ValueError:
    pass
  return '[unknown]'

class PrimopDiagnosticTool(BaselineTool):
  TOOL_NAME='_primop_diag'
  TOOL_SUMMARY='[INTERNAL] Assists in diagnosing Primop translation problems.'

  def add_subparser(self, argparser):
    super(PrimopDiagnosticTool, self).add_subparser(argparser)
    self.subparser.add_argument('--prioritized', '-p', action='store_true', default=False, help='Return a prioritized translation list based on the fraction of time spent in each native operation type.')
    return self.subparser

  def _run(self, modelfiles):
    self.data = dict()

    for modelfile in modelfiles:
      frame = FRAMEWORKS[self.args['framework']]()
      (modname, pypath) = path_to_loader_pair(modelfile)
      frame.load(RunpyLoader, modname, pypath=pypath)

      unknown_ops = dict()
      for primop in frame.absgraph:
        src = primop.source_op
        if primop.optype=='undef':
          if src.type not in unknown_ops:
            unknown_ops[src.type] = []
          args = []
          if src.op_def is not None: # None implies null-ary op
            for argdef,arg in zip(src.op_def.input_arg, src.inputs):
              s = str(argdef.name)+':'
              if type(arg)==tf.Tensor:
                s += 'T'+format_dims(arg)
              else:
                s += '(unknown type:'+str(type(argdef))+')'
              args.append(s)
          self.data[primop.id] = str(src.type)+'('+', '.join(args)+')'

      if self.args['prioritized']:
        raise NotImplementedError, 'Timing-prioritized diagnosis not available yet.'



  def _output(self):
    for k,v in self.data.items():
      print k,'=>',v

ToolRegistry.register(PrimopDiagnosticTool)
