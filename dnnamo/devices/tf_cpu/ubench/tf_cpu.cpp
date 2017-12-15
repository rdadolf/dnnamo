#include "ubench.hpp"
#include "tf_cpu.hpp"
#include "tf_cpu_ops.cpp"

#include <iostream>
#include <chrono>
#include <vector>
#include <algorithm>

#include "unsupported/Eigen/CXX11/Tensor"

#if 1
  // Simplistic model
  #define EIGEN_DONT_PARALLELIZE
#else
  // As in TensorFlow
  #define EIGEN_USE_THREADS
#endif

// TODO: use Tensors and unsupported Eigen operations, as in TensorFlow

using namespace Eigen;
using Eigen::Tensor;

// TODO: should setRandom be per trial?
double time_mmmul(std::vector<int> dim_A, std::vector<int> dim_B, size_t trials, size_t iterations)
{
  Tensor<float, 2, RowMajor> in0(dim_A[0], dim_A[1]);
  Tensor<float, 2, RowMajor> in1(dim_B[0], dim_B[1]);
  Tensor<float, 2, RowMajor> out(dim_A[0], dim_B[1]);

  in0.setRandom();
  in1.setRandom();

  // don't take transpose A of B (matmul_op.cc line 144)
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dims;
  dims[0].first = 1;
  dims[0].second = 0;

  // TESTED.
  ubench::State s = ubench::run_benchmark(
    [&out, &in0, &in1, &dims]() { 
      out = in0.contract(in1, dims);
    },
    trials,
    iterations
  );
  /* DEBUG
  std::cout << " A::\n";
  for( int i=0; i< dim_A[0]; i++){
    for( int j=0; j< dim_A[1]; j++){
      std::cout << in0(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << " B::\n";
  for( int i=0; i< dim_B[0]; i++){
    for( int j=0; j< dim_B[1]; j++){
      std::cout << in1(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << " Res::\n";
  for( int i=0; i< dim_A[0]; i++){
    for( int j=0; j< dim_B[1]; j++){
      std::cout << out(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "mmul test";
  */
  //return s.get_mean_duration();
  return s.get_mean_duration().count();
}

double time_mvmul(std::vector<int> dim_A, int dim_b, size_t trials, size_t iterations)
{
  std::cout << dim_b << "\n";
  Tensor<float, 2, RowMajor> in0(dim_A[0], dim_A[1]);
  Tensor<float, 1, RowMajor> in1(dim_b);
  Tensor<float, 1, RowMajor> out(dim_A[0]);

  in0.setRandom();
  in1.setRandom();

  // don't take transpose A of B (matmul_op.cc line 144)
  Eigen::array<Eigen::IndexPair<Eigen::DenseIndex>, 1> dims;
  dims[0].first = 0;
  dims[0].second = 1;

  // TESTED.
  ubench::State s = ubench::run_benchmark(
    [&out, &in0, &in1, &dims]() { 
      out = in1.contract(in0, dims);
    },
    trials,
    iterations
  );
  // DEBUG
  std::cout << " A::\n";
  for( int i=0; i< dim_A[0]; i++){
    for( int j=0; j< dim_A[1]; j++){
      std::cout << in0(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << " B::\n";
  std::cout << dim_b << "\n";
  for( int i=0; i< dim_b; i++){
    std::cout << in1(i) << " ";
  }
  std::cout << "\n";
  std::cout << " Res::\n";
  for( int i=0; i< dim_A[0]; i++){
      std::cout << out(i) << " ";
  }
  std::cout << "\n";
  std::cout << "mvmul test";

    return s.get_mean_duration().count();
}

double time_vvadd(int dim_a, int dim_b, size_t trials, size_t iterations) {
    VectorXd a(dim_a);
    a.setRandom();

    VectorXd b(dim_b);
    b.setRandom();

    VectorXd sum(dim_b);

    ubench::State s = ubench::run_benchmark(
        [&a, &b, &sum]() { 
            sum.noalias() += a + b;
        },
        trials,
        iterations
    );

    return s.get_mean_duration().count();
}


double time_conv(std::vector<int> dim_M, std::vector<int> dim_F, size_t trials, size_t iterations)
{

  const int input_depth = dim_M[0];
  const int input_rows  = dim_M[1];
  const int input_cols  = dim_M[2];

  const int patch_rows = dim_F[0];
  const int patch_cols = dim_F[1];

  const int output_depth = 1;
  const int output_rows = input_rows;
  const int output_cols = input_cols;

  Tensor<float, 3, RowMajor> input(input_cols, input_rows, input_depth);
  Tensor<float, 4, RowMajor> kernel(patch_cols, patch_rows, input_depth, output_depth);
  Tensor<float, 3, RowMajor> result(output_cols, output_rows, output_depth);

  input.setRandom();
  kernel.setRandom();
  result.setRandom();

  // Execute the Convolution and time it using the profiling harness
  // If we need to rip out the formating/templating we can..
  // but that should get taken out at compile time?
  // NOTE:: filter "falls off" doesn't wrap around..
  // TESTED.
  ubench::State s = ubench::run_benchmark(
      [&result, &input, &kernel]() {
        result = SpatialConvolution(input, kernel);
      }, trials, iterations);

  /* DEBUG
  std::cout << " Input::\n";
  for( int i=0; i< input_cols; i++){
    for( int j=0; j< input_rows; j++){
      std::cout << input(i, j, 0) << " ";
    }
    std::cout << "\n";
  }
  std::cout << " Kernel::\n";
  for( int i=0; i< patch_cols; i++){
    for( int j=0; j< patch_rows; j++){
      std::cout << kernel(i, j, 0, 0) << " ";
    }
    std::cout << "\n";
  }
  std::cout << " Res::\n";
  for( int i=0; i< output_cols; i++){
    for( int j=0; j< output_rows; j++){
      std::cout << result(i, j, 0) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "conv test";
  */  
  return s.get_mean_duration().count();
}

