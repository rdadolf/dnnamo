import tensorflow as tf

from ..framework import FRAMEWORKS
from ..loader import RunpyLoader
from .tool_utilities import BaselineTool, ToolRegistry

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
def dimstr_proto(dim_proto):
  if dim_proto.size==-1:
    return '?'
  else:
    return str(dim_proto.size)
def format_dims_proto(tf_shape_proto):
  return '['+'x'.join([dimstr_proto(d) for d in tf_shape_proto.dim])+']'

class PrimopDiagnosticTool(BaselineTool):
  TOOL_NAME='_primop_diag'
  TOOL_SUMMARY='[DEV] Assists in diagnosing Primop translation problems.'

  def add_subparser(self, argparser):
    super(PrimopDiagnosticTool, self).add_subparser(argparser)
    self.subparser.add_argument('--prioritized', '-p', action='store_true', default=False, help='Return a prioritized translation list based on the fraction of time spent in each native operation type.')
    return self.subparser

  def _run(self, models):
    self.data = dict()

    for model in models:
      frame = FRAMEWORKS[self.args['framework']]()
      frame.load(self.args['loader'], model, **self.args['loader_opts'])

      unknown_ops = dict()
      for primop in frame.get_absgraph():
        src = primop.source_op

        if primop.optype=='undef':
          if src.type not in unknown_ops:
            unknown_ops[src.type] = []
          args = []
          if src.op_def is not None:
            # None implies no Tensor dataflow inputs (config args only)
            for argdef,arg in zip(src.op_def.input_arg, src.inputs):
              s = str(argdef.name)+':'
              if type(arg)==tf.Tensor:
                s += 'T'+format_dims(arg)
              else:
                s += '(unknown type:'+str(type(argdef))+')'
              args.append(s)
          for attr_k, attr_v in src.node_def.attr.items():
            s = attr_k+'*:'
            attr_type = attr_v.WhichOneof('value')
            if attr_type=='s':
              s += '"'+str(attr_v.s)+'"'
            elif attr_type=='i':
              s += '"'+str(attr_v.i)+'"'
            elif attr_type=='f':
              s += '"'+str(attr_v.f)+'"'
            elif attr_type=='b':
              s += str(attr_v.b)
            elif attr_type=='type':
              s += 'type'
            elif attr_type=='tensor':
              s += 'T'+format_dims_proto(attr_v.tensor.tensor_shape)
            elif attr_type=='shape':
              s += 'T'+format_dims_proto(attr_v.shape)
            elif attr_type=='func':
              s += 'func'
            elif attr_type=='placeholder':
              s += 'place'
            elif attr_type=='list':
              # Should only be one, but leave it as if's not elif's, just in case
              if len(attr_v.list.s)>0:
                s += ','.join(['"'+str(_)+'"' for _ in attr_v.list.s])
              if len(attr_v.list.i)>0:
                s += ','.join(['"'+str(_)+'"' for _ in attr_v.list.i])
              if len(attr_v.list.f)>0:
                s += ','.join(['"'+str(_)+'"' for _ in attr_v.list.f])
              if len(attr_v.list.b)>0:
                s += ','.join([str(_) for _ in attr_v.list.b])
              if len(attr_v.list.type)>0:
                s += ','.join(['type' for _ in attr_v.list.type])
              if len(attr_v.list.shape)>0:
                s += ','.join(['T'+format_dims_proto(_) for _ in attr_v.list.shape])
              if len(attr_v.list.tensor)>0:
                s += ','.join(['T'+format_dims_proto(_.tensor_shape) for _ in attr_v.list.tensor])
              if len(attr_v.list.func)>0:
                s += ','.join(['"'+str(_)+'"' for _ in attr_v.list.func])

            args.append(s)
          
          self.data[primop.id] = str(src.type)+'('+', '.join(args)+')'

      if self.args['prioritized']:
        raise NotImplementedError, 'Timing-prioritized diagnosis not available yet.'



  def _output(self):
    for k,v in self.data.items():
      print k,'=>',v

ToolRegistry.register(PrimopDiagnosticTool)
