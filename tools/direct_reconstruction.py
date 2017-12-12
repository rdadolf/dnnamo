import nnmodel
import nnmodel.frameworks
import nnmodel.devices
from nnmodel.core.trace import average_traces, Tracepoint, Trace
from tool_utilities import *

class Tool(PlotTool):
  TOOL_NAME='direct_reconstruction'
  TOOL_SUMMARY='Attempt to mimic the performance behavior of a neural network by aggregating a set of primop microbenchmarks.'

  def __init__(self):
    super(Tool,self).__init__()
    self.data = []

  def add_subparser(self, argparser):
    super(Tool,self).add_subparser(argparser)
    self.subparser.add_argument('--device', default='tf_cpu', help='Specify the dvice to use as a target for the reconstruction.')
    return self.subparser

  def _run(self, modelfiles):
    for modelfile in modelfiles:
      frame = nnmodel.frameworks.FRAMEWORKS[self.args['framework']]()
      dev = nnmodel.devices.DEVICES[self.args['device']]()
      frame.load(modelfile)
      g = frame.graph()
      traces = frame.run_native_trace(n_steps=12)[1:-1] # avoid outliers
      trace = average_traces(traces)

      reconstruction = Trace()
      for _,tracepoint in enumerate(trace):
        primop_id = frame.translate_native_op(tracepoint.name)
        primop = g[primop_id]
        if primop.optype != 'undef':
          t = dev.run_benchmark(primop)
        else:
          t = 0
        reconstruction.append(Tracepoint(
          name=primop.id,
          type=primop.optype,
          device=primop.device,
          dt=t))

      assert len(trace)==len(reconstruction), 'Reconstruction is corrupted: native operations are missing.'
      for native_tp, recon_tp in zip(trace, reconstruction):
        self.data.append( [[native_tp.name, native_tp.type, native_tp.dt],[recon_tp.name, recon_tp.type, recon_tp.dt]] )

  def _output(self):
    super(Tool,self)._output()

    nat_sum,prim_sum = 0,0
    for (nat_name,nat_type,nat_dt),(prim_name,prim_type,prim_dt) in self.data:
      if prim_type != 'undef':
        print nat_name+'('+str(nat_type)+'):',nat_dt,',',prim_name+'('+prim_type+'):',prim_dt
      nat_sum += nat_dt
      prim_sum += prim_dt

    print ''
    print 'Native trace total:',nat_sum
    print 'Reconstructed total:',prim_sum

  def _plot(self, filename):
    (fig,ax) = plt.subplots(1,1)

    nats = []
    prims = []
    for (_,_,nat_dt),(_,_,prim_dt) in self.data:
      nats.append(nat_dt)
      prims.append(prim_dt)

    maxnum = max(max(nats),max(prims))
    ax.plot([0,maxnum],[0,maxnum],lw=0.5,c='gray')

    ax.scatter(nats, prims, facecolor=clr[0], lw=0)
    ax.set_xlim(0,maxnum)
    ax.set_ylim(0,maxnum)

    fig.tight_layout()
    fig.savefig(filename)
