import unittest
import subprocess

class TestUBench(unittest.TestCase):
  @classmethod
  def setUpClass(cls):
    subprocess.check_call('make -C nnmodel/devices/tf_cpu/ubench', shell=True, stderr=subprocess.STDOUT)

  def test_time_mmmul(self):
    import nnmodel.devices.tf_cpu.ubench.ubench as ubench
    import os
    print os.system('which python')
    # small debug test..
    #for (a,b,c) in [(4,2,3),]:
    for (a,b,c) in [(10,30,20), (100,300,200), (100,300,2000)]:
      dim_A = [a,b]
      dim_B = [b,c]
      t = ubench.time_mmmul(dim_A, dim_B, 1, 1)
      print 'MatMul took:',t

  def test_time_mvmul(self):
    import nnmodel.devices.tf_cpu.ubench.ubench as ubench
    import os
    print os.system('which python')
    # small debug test..
    #for (a,b,c) in [(4,2,3),]:
    for (a,b) in [(4,2),]:
      dim_A = [a,b]
      dim_b = b
      t = ubench.time_mvmul(dim_A, dim_b, 1, 1)
      print 'MatMul took:',t

  def test_time_conv(self):
    import nnmodel.devices.tf_cpu.ubench.ubench as ubench
    import os
    print os.system('which python')
    # This is the input image.
    # Expected to be in rowMajor.
    # dim[0] := input_depth
    # dim[1] := input_rows
    # dim[2] := input cols
    dim_M = [1, 500, 500]
    # This is the filter.
    # dim[0] := "patch_rows"
    # dim[1] := "patch_cols"
    dim_F = [5,5]
    trials = 1
    iterations = 1
    t = ubench.time_conv(dim_M, dim_F, trials, iterations)
    print 'Conv took:',t
