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
template <typename SrcT, typename DstT, bool Truncate>
static Graph* CastGraph(const string& kind, const TensorShape& shape) {
  auto* graph = new Graph(OpRegistry::Global());

  DataType src_type = DataTypeToEnum<SrcT>::v();
  DataType dst_type = DataTypeToEnum<DstT>::v();

  Tensor input_t(src_type, shape);
  input_t.flat<SrcT>().setRandom();
  Node* input = test::graph::Constant(graph, input_t, "input");
  const bool isDefault = (kind == "Default");

  Node* cast;
  Node* mklcast;
  if (isDefault) {
    TF_CHECK_OK(NodeBuilder(graph->NewName("Default_cast"), "Cast")
                    .Input(input)
                    .Attr("SrcT", src_type)
                    .Attr("DstT", dst_type)
                    .Attr("Truncate", false)
                    .Finalize(graph, &cast));

    return graph;
  }
  // Mkl Cast op.
  TF_CHECK_OK(NodeBuilder(graph->NewName("Mkl_cast"), "_MklCast")
                  .Input(input)
                  .Attr("SrcT", src_type)
                  .Attr("DstT", dst_type)
                  .Attr("Truncate", false)
                  .Finalize(graph, &mklcast));
  return graph;
}

#define BM_CAST(kind, A, B, type, SrcT, DstT)                           \
  static void BM_CAST_##kind##_##type##_##A##_##B##_##SrcT##_##DstT(    \
      ::testing::benchmark::State& state) {                             \
    int64 num_computed_elements = (A) * (B);                            \
    int64 flops_per_iter = num_computed_elements;                       \
                                                                        \
    test::Benchmark(#type, CastGraph<SrcT, DstT, false>(#kind, {A, B})) \
        .Run(state);                                                    \
    state.SetItemsProcessed(state.iterations() * flops_per_iter);       \
  }                                                                     \
  BENCHMARK(BM_CAST_##kind##_##type##_##A##_##B##_##SrcT##_##DstT)

#define BENCHMARK_CAST(A, B, type, SrcT, DstT) \
  BM_CAST(Default, A, B, type, SrcT, DstT);    \
  BM_CAST(Mkl, A, B, type, SrcT, DstT);

#define BENCHMARK_DTYPE(SrcT, DstT)              \
  BENCHMARK_CAST(0, 0, cpu, SrcT, DstT);         \
  BENCHMARK_CAST(0, 128, cpu, SrcT, DstT);       \
  BENCHMARK_CAST(512, 256, cpu, SrcT, DstT);     \
  BENCHMARK_CAST(1024, 512, cpu, SrcT, DstT);    \
  BENCHMARK_CAST(2048, 1024, cpu, SrcT, DstT);   \
  BENCHMARK_CAST(3072, 1024, cpu, SrcT, DstT);   \
  BENCHMARK_CAST(4096, 2048, cpu, SrcT, DstT);   \
  BENCHMARK_CAST(8192, 4096, cpu, SrcT, DstT);   \
  BENCHMARK_CAST(16384, 8192, cpu, SrcT, DstT);  \
  BENCHMARK_CAST(24576, 16384, cpu, SrcT, DstT); \
  BENCHMARK_CAST(32768, 16384, cpu, SrcT, DstT);

BENCHMARK_DTYPE(float, bfloat16)
BENCHMARK_DTYPE(bfloat16, float)
}  // namespace tensorflow

#endif  // INTEL_MKL
