from abc import ABCMeta, abstractmethod
#import json

class Device(object):
  __metaclass__ = ABCMeta

  @abstractmethod
  def run_benchmark(self, primop_type, *args): pass

  # FIXME
  #def read_config(self, filename):
  #  with open(filename, 'r') as f:
  #    conf = json.load(f)
  #    return conf
  #def write_config(self, conf, filename):
  #  with open(filename,'w') as f:
  #    json.dump(conf,f)

  # TODO:
  # Map memory load/store onto DGraph edges
  #   Need to show that bandwidth can be overlapped:
  #     for pieces that can, this is the steady-state bandwidth
  #       we'll need to balance this against sources and destinations
  #     for pieces that cannot, this is the load latency or "time-to-first-bit"
  #       we'll need to define a computational phase or synchronization block
  #   3 quantities then:
  #     ops, ss-b/w, ttfb
  # Map compute onto DGraph nodes
