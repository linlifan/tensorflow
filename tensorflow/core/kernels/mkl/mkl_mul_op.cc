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

using dnnl::engine;
using dnnl::stream;
using dnnl::memory;
using dnnl::algorithm;
using dnnl::binary;
using dnnl::primitive_attr;


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

template <typename T>
class MklNativeMul : public OpKernel {
 public:
  explicit MklNativeMul(OpKernelConstruction* context) : OpKernel(context) {
    // OP_REQUIRES_OK(context, context->GetAttr("SrcT", &src_dtype_));
    // OP_REQUIRES_OK(context, context->GetAttr("DstT", &dst_dtype_));
    // // printf("_MklCast OP Construction!\n");
  }

  virtual ~MklNativeMul() {}

  void Compute(OpKernelContext* context) override {
    try {
      // Grab the input tensor
      const Tensor& input_tensor_a = context->input(0);
      const Tensor& input_tensor_b = context->input(1);

      const bool input_a_is_scalar = TensorShapeUtils::IsScalar(input_tensor_a.shape());
      // TODO: Handle scalar tensors input_a
      assert(!input_a_is_scalar);

      const bool input_b_is_scalar = TensorShapeUtils::IsScalar(input_tensor_b.shape());

      // Handle scalar tensor input_b if input_b is scalar...
      Tensor input_tensor_bbc;
      if (input_b_is_scalar) {
        OP_REQUIRES_OK(context, context->allocate_temp(DataTypeToEnum<T>::v(),
                                                 input_tensor_a.shape(), &input_tensor_bbc));
        // TODO: allocate the value
      }


      // Handle empty tensors
      bool any_dim_zero = false;
      int dims = input_tensor_a.dims();
      any_dim_zero = (dims == 0 ? true : false);
      int64 input_num_elements = 1;
      for (int i = 0; i < dims; i++) {
        if (input_tensor_a.dim_size(i) == 0) any_dim_zero = true;
        input_num_elements *= input_tensor_a.dim_size(i);
      }
      

      // Create an output tensor
      Tensor* output_tensor = NULL;
      OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor_a.shape(),
                                                      &output_tensor));
      T* output = (T*)output_tensor->tensor_data().data();


      int current_thr_num = context->device()
                                ->tensorflow_cpu_worker_threads()
                                ->workers->AsEigenThreadPool()
                                ->NumThreads();
      // To estimate the proper number of threads to use
      // when given more available threads than you may need
      int estimate_thr_num = EstimateThreadsToUse(
          input_num_elements, sizeof(T), current_thr_num);

      // Create engine and stream
      auto cpu_engine = engine(engine::kind::cpu, 0);
      MklDnnData<T> in_a(&cpu_engine);
      MklDnnData<T> in_b(&cpu_engine);
      MklDnnData<T> out(&cpu_engine);
      std::shared_ptr<stream> cpu_stream;

      memory::dims src_a_dims = TFShapeToMklDnnDims(input_tensor_a.shape());
      memory::dims src_b_dims;
      if (input_b_is_scalar) {
        src_b_dims = TFShapeToMklDnnDims(input_tensor_bbc.shape());
      } else {
        src_b_dims = TFShapeToMklDnnDims(input_tensor_b.shape());
      }
      memory::dims src_a_strides = CalculateTFStrides(src_a_dims);
      memory::dims src_b_strides = CalculateTFStrides(src_b_dims);
      
      in_a.SetUsrMem(src_a_dims, src_a_strides, &input_tensor_a);
      if (input_b_is_scalar) {
        in_b.SetUsrMem(src_b_dims, src_b_strides, &input_tensor_bbc);
      } else {
        in_b.SetUsrMem(src_b_dims, src_b_strides, &input_tensor_b);
      }
      // Output dimensions are same as input dimensions.
      out.SetUsrMem(src_a_dims, src_a_strides, output_tensor);

      std::vector<primitive> prim_vec;
      std::vector<MemoryArgsMap> prim_args_vec;

      // Create operation descriptor.
      auto binary_d = binary::desc(algorithm::binary_mul, in_a.GetUsrMemDesc(), in_b.GetUsrMemDesc(), out.GetUsrMemDesc());

      MklDnnThreadPool eigen_tp(context, estimate_thr_num);

      engine cpu_engine_ = engine(engine::kind::cpu, 0);
      cpu_stream.reset(CreateStream(&eigen_tp, cpu_engine_));
      
      primitive_attr binary_attr;
      // Create primitive descriptor.
      auto binary_pd = binary::primitive_desc(binary_d, binary_attr, cpu_engine_);

      // Create the primitive.
      auto binary_prim = binary(binary_pd);


      in_a.SetUsrMemDataHandle(&input_tensor_a, cpu_stream);
      in_b.SetUsrMemDataHandle(&input_tensor_b, cpu_stream);
      out.SetUsrMemDataHandle(output_tensor, cpu_stream);

      prim_vec.push_back(binary_prim);


      // Primitive arguments. Set up in-place execution by assigning src_0 as DST.
      std::unordered_map<int, memory> binary_args;
      binary_args.insert({DNNL_ARG_SRC_0, *in_a.GetUsrMem()});
      binary_args.insert({DNNL_ARG_SRC_1, *in_b.GetUsrMem()});
      binary_args.insert({DNNL_ARG_DST, *out.GetUsrMem()});

      prim_args_vec.push_back(binary_args);

      execute_primitives(prim_vec, cpu_stream, prim_args_vec);

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
  // DataType src_dtype_;
  // DataType dst_dtype_;
  // bool use_truncation_ = false;
};


// register dnn kernels for supported operations and supported types
#define REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES(type)         \
  REGISTER_KERNEL_BUILDER(                                     \
      Name("_MklNativeMul")                                    \
          .Device(DEVICE_CPU)                                  \
          .TypeConstraint<type>("T")                          \
          .Label(mkl_op_registry::kMklNameChangeOpLabel),     \
      MklNativeMul<type>);                                     \
 

REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES(float)
REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES(bfloat16)

// TF_CALL_float(REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES);
// TF_CALL_bfloat16(REGISTER_MUL_MKL_SUPPORTED_KERNELS_TYPES);


}  // namespace tensorflow
#endif  // INTEL_MKL
