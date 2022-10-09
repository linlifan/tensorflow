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

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/core/common_runtime/kernel_benchmark_testlib.h"
#include "tensorflow/core/framework/fake_input.h"
#include "tensorflow/core/framework/node_def_builder.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/kernels/ops_testutil.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"

// Compare the performance of default tensorflow Cast kernel with
// _MklCast kernel on CPU.
//
// Then you could use below command to test _MklCast and default Cast
// performance:
// $ numactl -C xx-xx bazel run --action_env=TF_ENABLE_ONEDNN_OPTS=1 -c opt \
//  //tensorflow/core/kernels/mkl:mkl_cast_op_test -- --benchmark_filter=all
//
// Test with config MKL
// $ numactl -C xx-xx bazel run --config=mkl -c opt                         \
//  //tensorflow/core/kernels/mkl:mkl_cast_op_test -- --benchmark_filter=all

namespace tensorflow {

// --------------------------------------------------------------------------//
//           Test Cast performance VS _MklCast Kernel                        //
// --------------------------------------------------------------------------//

template <typename T>
static Graph* MulGraph(const string& kind, const TensorShape& shapeA, const TensorShape& shapeB) {
  auto* graph = new Graph(OpRegistry::Global());
  const bool isDefault = (kind == "Default");

  DataType data_type = DataTypeToEnum<T>::v();

  Tensor input_t_a(data_type, shapeA);
  Tensor input_t_b(data_type, shapeB);
  input_t_a.flat<T>().setRandom();
  input_t_b.flat<T>().setRandom(); 

  Node* input_a = test::graph::Constant(graph, input_t_a, "input_a");
  Node* input_b = test::graph::Constant(graph, input_t_b, "input_b");


  if (isDefault) {
    // TF_CHECK_OK(NodeBuilder(graph->NewName("Default_Mul"), "Mul")
    //                 .Input(input_a)
    //                 .Input(input_b)
    //                 .Attr("T", DT_FLOAT)
    //                 .Finalize(graph, nullptr));

    test::graph::Binary(graph, "Mul", input_a, input_b);
    return graph;
  }
  // Mkl Cast op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("Mkl_OneDNNMul"), "_MklNativeMul")
                  .Input(input_a)
                  .Input(input_b)
                  .Attr("T", data_type)
                  .Finalize(graph, nullptr));
  return graph; 
}

 
#define BM_Mul(kind, A, B, C, D, dt, type)                 \
  static void BM_##kind##_##type##_##A##_##B##_##C##_##D##_##dt(  \
      ::testing::benchmark::State& state) {                       \
    int64 num_computed_elements = (A) * (B);          \
    int64 flops_per_iter = num_computed_elements;                 \
                                                                  \
    if (D == 0) {                                                \ 
      test::Benchmark(#type, MulGraph<dt>(#kind, {A, B}, {}),     \
                    /*old_benchmark_api*/ false)                  \
        .Run(state);                                              \
    } else {                                                      \
    test::Benchmark(#type, MulGraph<dt>(#kind, {A, B}, {C, D}),  \
                    /*old_benchmark_api*/ false)                  \
        .Run(state);                                              \
    }                                                             \
    state.SetItemsProcessed(state.iterations() * flops_per_iter); \
  }                                                               \
  BENCHMARK(BM_##kind##_##type##_##A##_##B##_##C##_##D##_##dt)->UseRealTime();

#define BM(A, B, C, D, dt, type)                \
  BM_Mul(Default, A, B, C, D, dt, type); \
  BM_Mul(Mkl, A, B, C, D, dt, type); \


#define BM_FP32(A, B, C, D, type) \
  BM(A, B, C, D, float, type); \

#define BM_BF16(A, B, C, D, type) \
  BM(A, B, C, D, bfloat16, type); \

BM_FP32(512, 256, 512, 256, cpu)
BM_FP32(513, 257, 513, 257, cpu)
BM_FP32(1024, 256, 1024, 256, cpu)
BM_FP32(1025, 257, 1025, 257, cpu)
BM_FP32(256, 712, 256, 712, cpu)

BM_FP32(512, 256, 1, 0, cpu)
BM_FP32(513, 257, 1, 0, cpu)
BM_FP32(1024, 256, 1, 0, cpu)
BM_FP32(1025, 257, 1, 0, cpu)
BM_FP32(256, 712, 1, 0, cpu)

BM_BF16(512, 256, 512, 256, cpu)
BM_BF16(513, 257, 513, 257, cpu)
BM_BF16(1024, 256, 1024, 256, cpu)
BM_BF16(1025, 257, 1025, 257, cpu)
BM_BF16(256, 712, 256, 712, cpu)

BM_BF16(512, 256, 1, 0, cpu)
BM_BF16(513, 257, 1, 0, cpu)
BM_BF16(1024, 256, 1, 0, cpu)
BM_BF16(1025, 257, 1, 0, cpu)
BM_BF16(256, 712, 1, 0, cpu)

}  // namespace tensorflow

#endif  // INTEL_MKL
