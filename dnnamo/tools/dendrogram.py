from scipy.cluster import hierarchy

import dnnamo
from dnnamo.core.mpl_plot import *

from .tool_utilities import PlotTool, ToolRegistry

class DendrogramTool(PlotTool):
  TOOL_NAME='dendrogram'
  TOOL_SUMMARY='Plots a dendrogram for all considered models showing hierarchical clustering.'

  # make the names more readable
  rename_dict = {'ApplyGradientDescent': 'ApplySGD', 'SoftmaxCrossEntropyWithLogits': 'CrossEntropy',
                 'RandomStandardNormal': 'RandomNormal', 'AddN': 'Add', 'BiasAdd': 'Add', 'BatchMatMul': 'MatMul',
                 'Conv2DBackpropFilter': 'Conv2DBackFilter', 'Conv2DBackpropInput': 'Conv2DBackInput'}
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
    super(DendrogramTool,self).add_subparser(argparser)
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
      #frame.load(modelfile, device='/cpu:0', init_options={'batch_size':self.args['batch']})
      #frame.load(modelfile)
      traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True, 'inter_op_parallelism_threads':1, 'intra_op_parallelism_threads':8})
      #traces = frame.run_native_trace(n_steps=12, setup_options={'allow_soft_placement': True})
      # Hack off the first and last to avoid outliers.
      traces = traces[1:-1]
      self.data.append( [modelfile, self._aggregate_types(traces)] )

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
    for modelfile,data in self.data:
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
      new_self_data.append([modelfile,new_data])
      #new_traces.append([modelfile,{typ:dt for typ,dt in data.items() if dt>lowerbound}])
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
      for ((modelfile,data),before,after) in zip(self.data,n_op_types_before,n_op_types_after):
        name_time_dict = {}
        self.model_name_order.append(str(modelfile))
        print '# of op types for '+str(modelfile)+': '+str(before)+'->'+str(after)
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

  def rename_models(self, bad_names):
    names = ['alexnet', 'atari', 'vgg', 'residual', 'autoencoder', 'speech', 'memn2n', 'seq2seq']
    new_names = []
    print bad_names
    for m_name in bad_names:
      for name in names:
        if name in m_name:
          new_names.append(name)
          break
    return new_names

  def _plot(self, filename):
    (fig, ax) = plt.subplots(1, 1, figsize=(8, 5))

    plot_results = []
    for res_list in self.sorted_op_name_times:
      plot_results.append([float(xx) for xx in res_list])

    Z = hierarchy.linkage(plot_results, method='average', metric='cosine')
    dn = hierarchy.dendrogram(Z, orientation='right', color_threshold=0.0)

    dn['color_list'] = ['k',]*len(dn['color_list'])
    xt = ax.get_yticklabels()
    x_order = []
    for xx in xt:
      x_order.append(int(xx._text))
    real_labels = [self.model_name_order[xx] for xx in x_order]
    nice_names = self.rename_models(real_labels)
    real_x_labels = nice_names

    ax.set_yticklabels(real_x_labels, fontsize=24)
    for tick in ax.xaxis.get_major_ticks():
      tick.label.set_fontsize(20)

    ax.set_xlim(-.01, 1.01)

    ax.xaxis.set_ticks_position('bottom')

    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    fig.savefig('den.pdf', bbox_inches='tight')

ToolRegistry.register(DendrogramTool)
