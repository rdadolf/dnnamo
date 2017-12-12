#ifndef _TF_CPU_HPP_
#define _TF_CPU_HPP_

#include <iostream>
#include <vector>
#include <algorithm>

// TODO: move these elsewhere?
const size_t DEFAULT_TRIALS = 2;
const size_t DEFAULT_ITERATIONS = 10;

double time_mmmul(std::vector<int> dim_A, std::vector<int> dim_B, size_t trials = DEFAULT_TRIALS, size_t iterations = DEFAULT_ITERATIONS);
double time_mvmul(std::vector<int> dim_M, int dim_v, size_t trials = DEFAULT_TRIALS, size_t iterations = DEFAULT_ITERATIONS);
double time_vvadd(int dim_a, int dim_b, size_t trials = DEFAULT_TRIALS, size_t iterations = DEFAULT_ITERATIONS);
double time_conv(std::vector<int> dim_M, std::vector<int> dim_F, size_t trials = DEFAULT_TRIALS, size_t iterations = DEFAULT_ITERATIONS);

#endif//_TF_CPU_HPP_
