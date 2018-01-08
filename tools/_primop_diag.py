import dnnamo

from tool_utilities import BaselineTool, path_to_loader_pair

class Tool(BaselineTool):
  TOOL_NAME='_primop_diag'
  TOOL_SUMMARY='[INTERNAL] Assists in diagnosing Primop translation problems.'

  def add_subparser(self, argparser):
    super(Tool, self).add_subparser(argparser)
    self.subparser.add_argument('--prioritized', '-p', action='store_true', default=False, help='Return a prioritized translation list based on the fraction of time spent in each native operation type.')
    return self.subparser

  def _run(self, modelfiles):
    self.data = dict()

    for modelfile in modelfiles:
      frame = dnnamo.frameworks.FRAMEWORKS[self.args['framework']]()
      (modname, pypath) = path_to_loader_pair(modelfile)
      frame.load(dnnamo.loader.RunpyLoader, modname, pypath=pypath)

      #results = frame.analyze('abstractgraph',trigger='lazy')
      #results.absgraph

      if self.args['prioritized']:
        raise NotImplementedError, 'Timing-prioritized diagnosis not available yet.'



  def _output(self):
    for k,v in self.data.items():
      print k,'=>',v
