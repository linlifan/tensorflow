/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file uses OneDNN Reorder for acceleration of TF Cast,
// only for FP32TOBF16 and BF16TOFP32 operations.
// When Auto-Mixed Presicion insert 'Cast' Nodes, insert '_MklCast' instead.

#ifdef INTEL_MKL
#define EIGEN_USE_THREADS

#include <cmath>
#include "dnnl.hpp"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/mkl_util.h"
#include "tensorflow/core/kernels/cast_op_impl.h"
#include "tensorflow/core/kernels/cast_op.h"

using dnnl::engine;
using dnnl::stream;
using dnnl::memory;

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;

inline int EstimateThreadsToUse(int64 num_elements, int load_bytes,
                                int max_threads) {
  // We do this calculation to estimate the cost of a Cast op.
  // This func returns the proper number of threads in [1:max_threads] to use
  // with the given num_elements and cost per coefficient.

  // Cost of memory fetches from L2 cache.
  // 64 is typical cache line size.
  // 16 is L2 cache latency for SPR.
  const double kLoadCycles = 1.0 / 64 * 16;
  const double kStoreCycles = 1.0 / 64 * 16;
  // Scaling from Eigen compute cost to device cycles.
  static const int kDeviceCyclesPerComputeCycle = 1;
  // Costs in device cycles.
  static const int kStartupCycles = 100000;
  static const int kPerThreadCycles = 100000;
  bool src_is_fp32 = (load_bytes == 4) ? true : false;
  int store_bytes = src_is_fp32 ? 2 : 4;
  // cvt_fp32_to_bf16 - 14 compute cycles per 16 elements
  // cvt_bf16_to_fp32 - 4 compute cycles per 16 elements
  double compute_cycles = 1.0 * (src_is_fp32 ? 14 : 4) / 16;
  double totalCost =
      num_elements * (kLoadCycles * load_bytes + kStoreCycles * store_bytes +
                      kDeviceCyclesPerComputeCycle * compute_cycles);
  int threads = (totalCost - kStartupCycles) / kPerThreadCycles + 0.9;
  return std::fmin(max_threads, std::fmax(1, threads));
}

template <typename SrcT, typename DstT, bool Truncate>
class CastMkl : public OpKernel {
 public:
  explicit CastMkl(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("SrcT", &src_dtype_));
    OP_REQUIRES_OK(context, context->GetAttr("DstT", &dst_dtype_));
    // printf("_MklCast OP Construction!\n");
  }

  virtual ~CastMkl() {}

  void Compute(OpKernelContext* context) override {
    try {
      // Grab the input tensor
      const Tensor& input_tensor = context->input(0);
      SrcT* input = (SrcT*)input_tensor.tensor_data().data();

      // Handle scalar tensors
      const bool input_is_scalar = TensorShapeUtils::IsScalar(input_tensor.shape());

      // Handle empty tensors
      int dims = input_tensor.dims();
      bool any_dim_zero = (dims == 0 ? true : false);
      int64 input_num_elements = 1;
      for (int i = 0; i < dims; i++) {
        if (!any_dim_zero) {
          int64 per_dim_size = input_tensor.dim_size(i);
          input_num_elements *= per_dim_size;
          any_dim_zero = (per_dim_size == 0 ? true : false);
        }
      }

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                      &output_tensor));
      DstT* output = (DstT*)output_tensor->tensor_data().data();

      // Scalar and empty tensors should fall back to tf.cast
      if (input_is_scalar || any_dim_zero) {
        if (src_dtype_ == DT_FLOAT) {
          work_ = GetCpuCastFromFloat(dst_dtype_);
        } else {
          work_ = GetCpuCastFromBfloat(dst_dtype_);
        }
        work_(context, input_tensor, output_tensor, false);
        return;
      }

      int current_thr_num = context->device()
                                ->tensorflow_cpu_worker_threads()
                                ->workers->AsEigenThreadPool()
                                ->NumThreads();
      // To estimate the proper number of threads to use
      // when given more available threads than you may need
      int estimate_thr_num = EstimateThreadsToUse(
          input_num_elements, sizeof(SrcT), current_thr_num);

      // Create engine and stream
      auto cpu_engine = engine(engine::kind::cpu, 0);
      MklDnnData<SrcT> in(&cpu_engine);
      MklDnnData<DstT> out(&cpu_engine);
      std::shared_ptr<stream> cpu_stream;

      memory::dims src_dims = TFShapeToMklDnnDims(input_tensor.shape());
      memory::dims src_strides = CalculateTFStrides(src_dims);

      in.SetUsrMem(src_dims, src_strides, &input_tensor);
      // Output dimensions are same as input dimensions.
      out.SetUsrMem(src_dims, src_strides, output_tensor);

      std::vector<primitive> reorder_prim;
      auto* prim = FindOrCreateReorder<SrcT>(in.GetUsrMem(), out.GetUsrMem());

      MklDnnThreadPool eigen_tp(context, estimate_thr_num);
      cpu_stream.reset(CreateStream(&eigen_tp, prim->GetEngine()));

      in.SetUsrMemDataHandle(&input_tensor, cpu_stream);
      out.SetUsrMemDataHandle(output_tensor, cpu_stream);
      reorder_prim.push_back(*(prim->GetPrimitive()));

      std::vector<MemoryArgsMap> reorder_args;
      reorder_args.push_back(
          {{DNNL_ARG_FROM, *in.GetUsrMem()}, {DNNL_ARG_TO, *out.GetUsrMem()}});
      execute_primitives(reorder_prim, cpu_stream, reorder_args);
    } catch (dnnl::error& e) {
      string error_msg = "Status: " + std::to_string(e.status) + ", message: " +
                         string(e.message) + ", in file " + string(__FILE__) +
                         ":" + std::to_string(__LINE__);
      OP_REQUIRES_OK(
          context,
          errors::Aborted("Operation received an exception:", error_msg));
    }
  }

 protected:
  DataType src_dtype_;
  DataType dst_dtype_;
  // bool use_truncation_ = false;
  CastFunctorType work_ = nullptr;
};

REGISTER_KERNEL_BUILDER(Name("_MklCast")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<float>("SrcT")
                            .TypeConstraint<bfloat16>("DstT"),
                        CastMkl<float, bfloat16, false>);

REGISTER_KERNEL_BUILDER(Name("_MklCast")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<bfloat16>("SrcT")
                            .TypeConstraint<float>("DstT"),
                        CastMkl<bfloat16, float, false>);
}  // namespace tensorflow
#endif  // INTEL_MKL

