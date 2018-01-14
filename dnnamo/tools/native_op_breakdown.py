import numpy as np

from matplotlib.pyplot import cm
from matplotlib.colors import LogNorm

import dnnamo
from dnnamo.core.mpl_plot import *

from .tool_utilities import PlotTool, ToolRegistry

class NativeOpBreakdownTool(PlotTool):
  TOOL_NAME='native_op_breakdown'
  TOOL_SUMMARY='Computes a breakdown of the cumulative time spent in each native operation type.'

  # make the names more readable
  rename_dict = {'ApplyGradientDescent': 'ApplySGD',
                 'SoftmaxCrossEntropyWithLogits': 'CrossEntropy',
                 'RandomStandardNormal': 'RandomNormal',
                 'AddN': 'Add',
                 'BiasAdd': 'Add',
                 'BatchMatMul': 'MatMul',
                 'Conv2DBackpropFilter': 'Conv2DBackFilter',
                 'Conv2DBackpropInput': 'Conv2DBackInput'}
  # order of Ops in the table/plot includes categories
  table_list_order = [ 'Add', 'Sub', 'Mul', 'Div', 'Pow', 'Softmax', #Elementwise
                       'MatMul', #Matrix
                       'CrossEntropy', 'MaxPoolGrad', 'Sum', #Reduction
                       'Conv2D', 'Conv2DBackFilter', 'Conv2DBackInput', #Conv.
                       'RandomNormal', #Reg.
                       'ApplyAdam', 'ApplyRMSProp', #Optimization
                       'Transpose', 'Tile', # data mov
                       'Select', 'Pad', 'Reshape', 'Shape', 'Concat',
                       'Variable', #sync
                     ]
  model_op_name_time = []
  model_name_order = []
  sorted_op_name_times = []

  def add_subparser(self, argparser):
    super(NativeOpBreakdownTool,self).add_subparser(argparser)
    self.subparser.add_argument('--print', action='store_true', default=False, help='Print a sorted list of the most time-consuming native operations.')
    self.subparser.add_argument('--threshold', action='store', type=float, default=101, help='Only consider the most time-consuming operations whose cumulative runtime is above a certain percentage of the total.')
    # comment this out
    # self.subparser.add_argument('--batch', action='store', type=int, default=32, help='Pass and sweep the batch size.')
    return self.subparser

  def _run(self, models):
    self.data = []
    for model in models:
      frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]()
      # pass batch_size in here.. also add a command line for it.
      frame.load(self.args['loader'], model, **self.args['loader_opts'])
      #frame.load(model, device='/cpu:0', init_options={'batch_size':self.args['batch']})
      traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True, 'inter_op_parallelism_threads':1, 'intra_op_parallelism_threads':8})
      #traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True})
      # Hack off the first and last to avoid outliers.
      traces = traces[1:-1]
      self.data.append( [model, self._aggregate_types(traces)] )

  def _aggregate_types(self, traces):
    breakdown = {} # op -> microseconds
    op_platform = {} # op -> microseconds
    for trace in traces:
      for op in trace:
        typ,dt,device = op.type,op.dt,op.device
        if typ not in breakdown:
          breakdown[typ] = 0
        breakdown[typ] += dt
        op_platform[typ] = device
    print op_platform.items()
    return breakdown

  def _threshold(self):
    new_self_data = []
    for model,data in self.data:
      total_time = sum(data.values())
      print 'total_time',total_time
      threshold = total_time*self.args['threshold']/100.
      print 'threshold',threshold
      sorted_data = sorted(data.items(), key=lambda p:p[1], reverse=True)
      new_data = {}
      s = 0
      for typ,dt in sorted_data:
        if s>=threshold:
          break
        new_data[typ]=dt
        s += dt
      new_self_data.append([model,new_data])
      #new_traces.append([model,{typ:dt for typ,dt in data.items() if dt>lowerbound}])
    self.data = new_self_data

  def sort_results(self):
    ii = 0
    table_dict = {}
    for td in self.table_list_order:
      table_dict[td] = ii
      ii += 1

    for mont in self.model_op_name_time:
      temp_times =[0]*(len(self.table_list_order))
      for n in mont:
        temp_times[table_dict[n]] = mont[n]
      self.sorted_op_name_times.append(temp_times)

  def _output(self):
    n_op_types_before = [len(data[1]) for data in self.data]
    self._threshold()
    n_op_types_after = [len(data[1]) for data in self.data]
    if self.args['print']:
      for ((model,data),before,after) in zip(self.data,n_op_types_before,n_op_types_after):
        name_time_dict = {}
        self.model_name_order.append(str(model))
        print '# of op types for '+str(model)+': '+str(before)+'->'+str(after)
        for optype,dt in sorted(data.items(), key=lambda p:p[1], reverse=True):
          print '  '+str(optype)+': '+str(dt)
          # NOTE: this is where renaming happens..
          try:
            name_time_dict[self.rename_dict[optype]] = dt
          except:
            name_time_dict[optype] = dt
        self.model_op_name_time.append(name_time_dict)
    self.sort_results()

    if not self.args['noplot']:
      self._plot(self.args['plotfile'])

  def rename_models(self):
    names = ['alexnet', 'atari', 'vgg', 'residual', 'autoencoder', 'speech', 'memn2n', 'seq2seq']
    new_names = []
    for m_name in self.model_name_order:
      for name in names:
        if name in m_name:
          new_names.append(name)
          break
    return new_names

  def _plot(self, filename):
    fnt_size = 5
    (_, ax) = plt.subplots(1, 1, figsize=(3,8))

    nice_names = self.rename_models()

    norm_results = []
    for res_list in self.sorted_op_name_times:
      norm_results.append([float(xx)/float(sum(res_list)) for xx in res_list])

    for y in range(len(norm_results)):
      for x in range(len(norm_results[0])):
        plt.text(x + 0.5 , y + 0.45, '%d' % int(100*norm_results[y][x]),
                  horizontalalignment='center', verticalalignment='center',
                  color='k', fontsize=fnt_size)

    dy = 1
    dx = 1
    y, x = np.mgrid[slice(0, len(norm_results) + dy, dy),
                    slice(0, len(norm_results[0]) + dx, dx)]
    cmap=cm.Reds
    plt.pcolor(x, y, norm_results, alpha=0.75, linestyle='solid',edgecolors='k',
                  norm=LogNorm(vmin=0.01, vmax=0.90), cmap=cmap)

    plt.yticks(np.arange(0.5, len(norm_results), 1), nice_names)
    plt.xticks(np.arange(0.5, len(self.table_list_order), 1), self.table_list_order, rotation='vertical')
    plt.tick_params(axis='both', which='both', left='off', right='off',bottom='off', top='off')

    ax.set_aspect('equal')
    ax.set_xlim(0, len(self.table_list_order))

    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(fnt_size)
    for tick in ax.yaxis.get_major_ticks():
      tick.label.set_fontsize(fnt_size)
    plt.savefig('heat.pdf', bbox_inches='tight')

ToolRegistry.register(NativeOpBreakdownTool)
