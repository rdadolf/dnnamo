
class Solver(object):
  def __init__(self, dgraph, devicemap):
    self.graph = dgraph
    self.devicemap = devicemap

  def eval(self, variable):
    return self.graph.eval(variable)
