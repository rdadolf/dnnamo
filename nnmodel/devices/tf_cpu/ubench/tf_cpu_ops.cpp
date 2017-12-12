#include "ubench.hpp"
#include "tf_cpu.hpp"

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

// Spatial Convolution from Tensor Flow
template <typename Input, typename Kernel>
EIGEN_ALWAYS_INLINE static const typename internal::conditional<
    internal::traits<Input>::Layout == ColMajor,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorImagePatchOp<Dynamic, Dynamic, const Input> > > >,
    TensorReshapingOp<
        const DSizes<typename internal::traits<Input>::Index,
                     internal::traits<Input>::NumDimensions>,
        const TensorContractionOp<
            const array<IndexPair<typename internal::traits<Input>::Index>, 1>,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const TensorImagePatchOp<Dynamic, Dynamic, const Input> >,
            const TensorReshapingOp<
                const DSizes<typename internal::traits<Input>::Index, 2>,
                const Kernel> > > >::type
SpatialConvolution(const Input& input, const Kernel& kernel,
                   const DenseIndex row_stride = 1,
                   const DenseIndex col_stride = 1,
                   const PaddingType padding_type = PADDING_SAME,
                   const DenseIndex row_in_stride = 1,
                   const DenseIndex col_in_stride = 1) {
  typedef typename internal::traits<Input>::Index TensorIndex;
  TensorRef<Tensor<typename internal::traits<Input>::Scalar,
                   internal::traits<Input>::NumDimensions,
                   internal::traits<Input>::Layout, TensorIndex> >
      in(input);
  TensorRef<Tensor<typename internal::traits<Kernel>::Scalar,
                   internal::traits<Kernel>::NumDimensions,
                   internal::traits<Kernel>::Layout, TensorIndex> >
      kern(kernel);

  EIGEN_STATIC_ASSERT(
      internal::traits<Input>::Layout == internal::traits<Kernel>::Layout,
      YOU_MADE_A_PROGRAMMING_MISTAKE);
  static const bool isColMajor = (internal::traits<Input>::Layout == ColMajor);

  static const int NumDims = internal::traits<Input>::NumDimensions;

  // Number of filters to apply. This is the same as the output depth of the
  // result
  const TensorIndex kernelFilters =
      isColMajor ? kern.dimensions()[0] : kern.dimensions()[3];
  // Number of channels. This is the same as the input depth.
  const TensorIndex kernelChannels =
      isColMajor ? kern.dimensions()[1] : kern.dimensions()[2];
  const TensorIndex kernelRows =
      isColMajor ? kern.dimensions()[2] : kern.dimensions()[1];
  const TensorIndex kernelCols =
      isColMajor ? kern.dimensions()[3] : kern.dimensions()[0];

  const DenseIndex kernelRowsEff =
      kernelRows + (kernelRows - 1) * (row_in_stride - 1);
  const DenseIndex kernelColsEff =
      kernelCols + (kernelCols - 1) * (col_in_stride - 1);

  array<IndexPair<TensorIndex>, 1> contract_dims;
  contract_dims[0] = IndexPair<TensorIndex>(1, 0);

  const TensorIndex InputRows =
      isColMajor ? in.dimension(1) : in.dimension(NumDims - 2);
  const TensorIndex InputCols =
      isColMajor ? in.dimension(2) : in.dimension(NumDims - 3);

  TensorIndex out_height;
  TensorIndex out_width;
  switch (padding_type) {
    case PADDING_VALID:
      out_height = numext::ceil((InputRows - kernelRowsEff + 1.f) /
                                static_cast<float>(row_stride));
      out_width = numext::ceil((InputCols - kernelColsEff + 1.f) /
                               static_cast<float>(col_stride));
      break;
    case PADDING_SAME:
      out_height = numext::ceil(InputRows / static_cast<float>(row_stride));
      out_width = numext::ceil(InputCols / static_cast<float>(col_stride));
      break;
    default:
      eigen_assert(false && "unexpected padding");
  }

  // Molds the output of the patch extraction code into a 2d tensor:
  // - the first dimension (dims[0]): the patch values to be multiplied with the
  // kernels
  // - the second dimension (dims[1]): everything else
  DSizes<TensorIndex, 2> pre_contract_dims;
  if (isColMajor) {
    pre_contract_dims[0] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[1] = out_height * out_width;
    for (int i = 3; i < NumDims; ++i) {
      pre_contract_dims[1] *= in.dimension(i);
    }
  } else {
    pre_contract_dims[1] = kernelChannels * kernelRows * kernelCols;
    pre_contract_dims[0] = out_height * out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      pre_contract_dims[0] *= in.dimension(i);
    }
  }

  // Molds the output of the contraction into the shape expected by the used
  // (assuming this is ColMajor):
  // - 1st dim: kernel filters
  // - 2nd dim: output height
  // - 3rd dim: output width
  // - 4th dim and beyond: everything else including batch size
  DSizes<TensorIndex, NumDims> post_contract_dims;
  if (isColMajor) {
    post_contract_dims[0] = kernelFilters;
    post_contract_dims[1] = out_height;
    post_contract_dims[2] = out_width;
    for (int i = 3; i < NumDims; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  } else {
    post_contract_dims[NumDims - 1] = kernelFilters;
    post_contract_dims[NumDims - 2] = out_height;
    post_contract_dims[NumDims - 3] = out_width;
    for (int i = 0; i < NumDims - 3; ++i) {
      post_contract_dims[i] = in.dimension(i);
    }
  }

  DSizes<TensorIndex, 2> kernel_dims;
  if (isColMajor) {
    kernel_dims[0] = kernelFilters;
    kernel_dims[1] = kernelChannels * kernelRows * kernelCols;
  } else {
    kernel_dims[0] = kernelChannels * kernelRows * kernelCols;
    kernel_dims[1] = kernelFilters;
  }
  // TODO(yangke): choose() is defined in TensorContraction.h -- consider
  // moving it to somewhere more "common".
  return choose(
      Cond<internal::traits<Input>::Layout == ColMajor>(),
      kernel.reshape(kernel_dims)
          .contract(input
                        .extract_image_patches(
                            kernelRows, kernelCols, row_stride, col_stride,
                            row_in_stride, col_in_stride, padding_type)
                        .reshape(pre_contract_dims),
                    contract_dims)
          .reshape(post_contract_dims),
      input
          .extract_image_patches(kernelRows, kernelCols, row_stride, col_stride,
                                 row_in_stride, col_in_stride, padding_type)
          .reshape(pre_contract_dims)
          .contract(kernel.reshape(kernel_dims), contract_dims)
          .reshape(post_contract_dims));
}
