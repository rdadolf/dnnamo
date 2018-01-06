
class Solver(object):
  def __init__(self, absgraph, devicemap):
    self._absgraph = absgraph
    self._devicemap = devicemap

  def eval(self, variable):
    return self.graph.eval(variable)
