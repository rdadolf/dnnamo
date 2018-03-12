
from .tool_utilities import AbstractTool, ToolRegistry

class SweepTool(AbstractTool):
  TOOL_NAME='sweep'
  TOOL_SUMMARY='Generating timing measurements suitable for training a performance estimator.'

  def add_subparser(self, argparser):
    super(SweepTool, self).add_subparser(argparser)
    self.subparser.add_argument('primop', type=str, help='Which primitive operation to sweep.')
    self.subparser.add_argument('--seed', type=int, help='Set the PRNG seed for this sweep.')
    # FIXME: Add argument to limit scope of sweep
    return self.subparser

  def _run(self):
    print 'Running'

  def _output(self):
    print 'Output'

ToolRegistry.register(SweepTool.TOOL_NAME, SweepTool)
