class Tracepoint(object):
  def __init__(self, name, type, device, dt):
    self.name = name
    self.type = type
    self.device = device
    self.dt = dt
  def __str__(self):
    return '<Tracepoint: name='+str(self.name)+' type='+str(self.type)+' device='+str(self.device)+' dt='+str(self.dt)+'>'

class Trace(object):
  def __init__(self):
    self._tracepoints = []
    self._namemap = {}
    self._len = 0
  def append(self, tracepoint):
    self._namemap[tracepoint.name] = self._len
    self._len += 1
    self._tracepoints.append(tracepoint)
  def __contains__(self, name):
    return name in self._namemap
  def __len__(self):
    return self._len
  def __iter__(self):
    for tracepoint in self._tracepoints:
      yield tracepoint
  def __getitem__(self, x):
    if isinstance(x, int):
      return self._tracepoints[x]
    elif isinstance(x, str):
      return self._tracepoints[self._namemap[x]]
    elif isinstance(x, slice):
      return self._tracepoints[x]
    raise TypeError('Invalid key "'+str(x)+'", must be integer or string')

def average_traces(traces):
  '''Average the time deltas across a set of traces. Start time becomes meaningless.'''
  mean_trace = Trace()

  # Sum all durations
  for trace in traces:
    for tracepoint in trace:
      name = tracepoint.name
      typ = tracepoint.type
      dev = tracepoint.device
      dt = tracepoint.dt
      if name not in mean_trace:
        mean_trace.append(Tracepoint(name,typ,dev,0))
      tp = mean_trace[name]
      assert tp.name==name, 'Name conflict during trace averaging: "'+str(tp.name)+'" vs "'+str(name)
      assert tp.type==typ, 'Type conflict during trace averaging: "'+str(tp.type)+'" vs "'+str(typ)
      assert tp.device==dev, 'Device conflict during trace averaging: "'+str(tp.device)+'" vs "'+str(dev)
      tp.dt += dt
  # Divide by number of traces
  n_traces = len(traces)
  for tracepoint in mean_trace:
    tracepoint.dt /= float(n_traces)

  return mean_trace

