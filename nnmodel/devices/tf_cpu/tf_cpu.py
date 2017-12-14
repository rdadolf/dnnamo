from nnmodel.core.device import Device
#from .ubench import ubench

class TF_CPU(Device):
  pass
  def run_benchmark(self,primop):
    return 0 # FIXME

  #  benchmark_name = 'time_'+str(primop.optype)
  #  benchmark_args = list(primop.args)+[10,10]

  #  assert hasattr(ubench,benchmark_name), 'No microbenchmark named "'+str(benchmark_name)+'" is implemented for device "tf_cpu"'
  #  t = getattr(ubench,benchmark_name)(*benchmark_args)
  #  return t
