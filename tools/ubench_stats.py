import dnnamo
import dnnamo.devices

TOOL_SUMMARY='Computes various statistics on device microbenchmarks.'

class Tool(object):
  def __init__(self):
    self.args = None

  def add_subparser(self, argparser):
    subparser = argparser.add_parser('ubench_stats', help=TOOL_SUMMARY)
    subparser.add_argument('--device', choices=dnnamo.devices.DEVICES.keys(), default='tf_cpu', help='choose a device to benchmark')
    return subparser

  def run(self, args):
    self.args = args
    # TODO
