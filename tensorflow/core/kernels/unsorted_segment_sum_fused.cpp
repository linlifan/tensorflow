#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

using namespace tensorflow;

// input: input tensor value (it sores the id)
// cols: How many elements to do SparseSegmentSum
// output: rows * embedding_size
template<typename T>
static void sparse_gather_v1(T *input, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    for (int j = 0; j < embedding_size; ++j) {
      float value = 0;
      int dense_num = 0;
      for (int k = 0; k < cols; ++k) {
        int embedding_row = (int)pidx[k];
        if (embedding_row >= 0) {
          value += embedding_table[embedding_row * embedding_size + j];
          dense_num += 1;
        }
      }

      if (is_mean && dense_num > 0) {
        *output++ = value / dense_num;
      } else {
        *output++ = value;
      }
    }
    pidx += cols;
  }
}

// embedding_size = 1
template<typename T>
static void sparse_gather_embeddingsize1(T *input, int rows, int cols, float *embedding_table, float *output, bool is_mean) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    float value = 0;
    int dense_num = 0;
    for (int k = 0; k < cols; ++k) {
      int embedding_row = pidx[k];
      if (embedding_row >= 0) {
        value += embedding_table[embedding_row];
        dense_num += 1;
      }
    }
    if (is_mean && dense_num > 0) {
      *output++ = value / dense_num;
    } else {
      *output++ = value;
    }
    pidx += cols;
  }
}

// input cols = 1
template<typename T>
static void sparse_gather_column1(T *input, int rows, float *embedding_table, float *output, int embedding_size) {
  T *pidx = input;
  for (int i = 0; i < rows; ++i) {
    int embedding_row = *pidx++;
    if (embedding_row >= 0) {
      float *pembedding = &embedding_table[embedding_row * embedding_size];
      for (int j = 0; j < embedding_size; ++j) {
        output[j] = pembedding[j];
      }
    } else {
      for (int j = 0; j < embedding_size; ++j) {
        output[j] = 0;
      }
    }
    output += embedding_size;
  }
}

template<typename T>
static void sparse_gather(T *input, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  if (embedding_size == 1) {
    sparse_gather_embeddingsize1(input, rows, cols, embedding_table, output, is_mean);
  } else if (cols == 1) {
    sparse_gather_column1(input, rows, embedding_table, output, embedding_size);
  } else {
    //printf("General sparse gather!\n");
    sparse_gather_v1(input, rows, cols, embedding_table, output, embedding_size, is_mean);
  }
}

// Use memcpy or manually assign?
static void mycopy(float *dst, float *src, int float_num) {
  memcpy(dst, src, float_num * sizeof(float));
}

static void myadd(float *dst, float *src, int float_num) {
  for (int i = 0; i < float_num; ++i) {
    dst[i] += src[i];
  }
}

static void myscale(float *dst, float factor, int float_num) {
  for (int i = 0; i < float_num; ++i) {
    dst[i] *= factor;
  }
}

template<typename Tid, typename Tshape>
static void sparse_gather(Tid *input, int64 input_size, Tshape *indice, int indice_dim, Tshape *shape, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean) {
  // Record how many values in each row
  int *row_values = new int[rows];
  memset(row_values, 0, rows * sizeof(int));

  for (int64 i = 0; i < input_size; ++i) {
    Tid id = input[i];
    if (id < 0) { // Skip invalid id
      continue;
    }
    auto row = indice[i * indice_dim];
    for (int k = 1; k < indice_dim - 1; ++k) {
      row = row * shape[k] + indice[i * indice_dim + k];
    }
    if (row_values[row] > 0) {
      myadd(&output[row * embedding_size], &embedding_table[id * embedding_size], embedding_size);
    } else {
      mycopy(&output[row * embedding_size], &embedding_table[id * embedding_size], embedding_size);
    }
    row_values[row] += 1;
  }

  for (int i = 0; i < rows; ++i) {
    if (row_values[i] == 0) {
      memset(&output[i * embedding_size], 0, embedding_size * sizeof(float));
    } else if (is_mean && row_values[i] > 1) {
      float factor = 1.0f / row_values[i];
      myscale(&output[i * embedding_size], factor, embedding_size);
    }
  }

  delete[] row_values;
}

REGISTER_OP("UnsortedSegmentSumFused")
    .Input("input0: Tdata")
    .Input("input1: Tid")
    //.Input("dense_shape: Tshape")
    //.Input("indice: Tshape")
    .Output("output: float")
    .Attr("Tdata: numbertype")
    .Attr("Tid: numbertype");
    //.Attr("Combiner: int")
    //.Attr("Tid: {int64, int32}")
    //.Attr("Tshape: {int64, int32}")
    //.Attr("Tweight: {float, resource}");
    /*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      DataType datatype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tweight", &datatype));
      if (datatype == DT_RESOURCE) {
        // TODO: Get the real shape
        c->set_output(0, c->Matrix(c->UnknownDim(), c->UnknownDim()));
      } else {
        ::tensorflow::shape_inference::ShapeHandle weight;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &weight));
        c->set_output(0, c->Matrix(c->UnknownDim(),
                                   c->Dim(c->input(0), 1)));
      }
      return Status::OK();
    });*/


template<typename Tid>
class UnsortedSegmentSumFusedOp : public OpKernel {
public:
  explicit UnsortedSegmentSumFusedOp(OpKernelConstruction* context) : OpKernel(context) {
    //OP_REQUIRES_OK(context, context->GetAttr("Combiner", &combiner));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));
    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** UnsortedSegmentSumFused ********\n");
      printed = true;
    }
  }

  ~UnsortedSegmentSumFusedOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Grab the weight
    float *data;
    const Tensor& data_tensor = context->input(0);
#if 0
    if (weight_tensor->dtype() == DT_RESOURCE) {
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), &variable));
      core::ScopedUnref s(variable);
      weight_tensor = variable->tensor();
      OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ", node_name));
    }
#endif

    data = (float *)data_tensor.tensor_data().data();
    
    // Input id
    const Tensor& segment_ids_tensor = context->input(1);
    Tid *segment_ids = (Tid *)segment_ids_tensor.tensor_data().data();
    

    // To check the input
    OP_REQUIRES(context, (data_tensor.dims() == 2),
	        errors::InvalidArgument("data tensor is not valid (dims != 2)"));
    OP_REQUIRES(context, (segment_ids_tensor.dims() == 1),
                errors::InvalidArgument("segment_ids tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (segment_ids_tensor.dim_size(0) == data_tensor.dim_size(0)),
               errors::InvalidArgument("segment ids tensor size not equal to data tensor size!"));

    int64 input_size = data_tensor.dim_size(0);
    Tid num_segments = 0;
    
    if (segment_ids_tensor.dim_size(0) > 100)
    {
        std::cout << "\n segment_ids_tensor.dims():" <<segment_ids_tensor.dims()<<", segment_ids_tensor.dim_size(0): "<< segment_ids_tensor.dim_size(0) << std::endl;
    }

    for (uint32_t i = 0; i < segment_ids_tensor.dim_size(0); i ++)
    {
	if (num_segments < segment_ids[i])
	{
            num_segments = segment_ids[i]; 
	}	
    }    
    //last index + 1 for number
    num_segments++;

    OP_REQUIRES(context, (num_segments > 0),
                errors::InvalidArgument("segment_ids number is not valid (num > 0)"));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({num_segments, 1});
    
    if (num_segments > 100)
    { 
        std::cout<<"num_segments: " << num_segments << std::endl;
    } 
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();
    
    for (uint32_t i = 0; i < output_tensor->dim_size(0); i++)
    {
        output[i] = 0;	    
    }

    for (uint32_t i = 0; i < segment_ids_tensor.dim_size(0); i++)
    {
        output[segment_ids[i]] += data[i];
    } 

  }

private:
  std::string node_name;
};



REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSumFused")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tid"),
                        UnsortedSegmentSumFusedOp<int32>);
REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSumFused")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tid"),
                        UnsortedSegmentSumFusedOp<int64>);


REGISTER_OP("UnsortedSegmentSumFused2")
    .Input("input0: Tdata")
    .Input("input1: Tid")
    .Output("output: float")
    .Attr("Tdata: numbertype")
    .Attr("Tid: numbertype");
   /*.SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      DataType datatype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tweight", &datatype));
      if (datatype == DT_RESOURCE) {
        // TODO: Get the real shape
        c->set_output(0, c->Matrix(c->UnknownDim(), c->UnknownDim()));
      } else {
        ::tensorflow::shape_inference::ShapeHandle weight;
        TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &weight));
        c->set_output(0, c->Matrix(c->UnknownDim(),
                                   c->Dim(c->input(0), 1)));
      }
      return Status::OK();
    });*/


template<typename Tid>
class UnsortedSegmentSumFused2Op : public OpKernel {
public:
  explicit UnsortedSegmentSumFused2Op(OpKernelConstruction* context) : OpKernel(context) {
    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** UnsortedSegmentSumFused2 ********\n");
      printed = true;
    }
  }

  ~UnsortedSegmentSumFused2Op() {
  }

  void Compute(OpKernelContext* context) override {
    float *data;
    
    const Tensor& data_tensor = context->input(0);
    const Tensor& segment_ids_tensor = context->input(1);

    // To check the input
    OP_REQUIRES(context, (data_tensor.dims() == 2),
	        errors::InvalidArgument("data tensor is not valid (dims != 2)"));
    OP_REQUIRES(context, (segment_ids_tensor.dims() == 1),
                errors::InvalidArgument("segment_ids tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (segment_ids_tensor.dim_size(0) == data_tensor.dim_size(0)),
               errors::InvalidArgument("segment ids tensor size not equal to data tensor size!"));
    
    auto segment_ids = segment_ids_tensor.flat<Tid>();
    for (uint32_t i = 0; i < segment_ids.size(); i++)
    {
        OP_REQUIRES(context, (segment_ids(i) == i), 
		    errors::InvalidArgument("segment ids not unique!"));
    }

    context->set_output(0, context->input(0));

  }

private:
  std::string node_name;
};



REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSumFused2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tid"),
                        UnsortedSegmentSumFused2Op<int32>);
REGISTER_KERNEL_BUILDER(Name("UnsortedSegmentSumFused2")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tid"),
                        UnsortedSegmentSumFused2Op<int64>);
