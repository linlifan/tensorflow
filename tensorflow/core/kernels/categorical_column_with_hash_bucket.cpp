
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

#include "tensorflow/core/platform/fingerprint.h"

using namespace tensorflow;

REGISTER_OP("CategoricalColumnWithHashBucket")
  .Input("input: Tid")
  .Input("weight: Tweight")
  .Output("hash: int64")
  .Output("indices: int64")
  .Output("dense_shape: int64")
  .Attr("Tid: {string, int64}")
  .Attr("Tweight: {string, int64}")
  .Attr("num_buckets: int >= 1")
  .Attr("feature_column_type: string")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      DataType datatype, weighttype;
      TF_RETURN_IF_ERROR(c->GetAttr("Tid", &datatype));
      TF_RETURN_IF_ERROR(c->GetAttr("Tweight", &weighttype));
      if ( datatype != weighttype )
        return errors::InvalidArgument("`input` must be the same type as `weight`.");

      std::string feature_column_type;
      TF_RETURN_IF_ERROR(c->GetAttr("feature_column_type", &feature_column_type));
      if ( feature_column_type != "CategoricalColumnWithHashBucket" )
        return errors::InvalidArgument("`feature_column_type` must be one of `CategoricalColumnWithHashBucket`.");
      c->set_output(0, c->Vector(c->UnknownDim()));
      c->set_output(1, c->Matrix(c->UnknownDim(), 2));
      c->set_output(2, c->Vector(c->UnknownDim()));

      return Status::OK();
    });


template<typename Tid, typename Tweight>
class CategoricalColumnWithHashBucketOp : public OpKernel {
public:
  explicit CategoricalColumnWithHashBucketOp(OpKernelConstruction* context) : OpKernel(context) {
    node_name = context->def().name();
    OP_REQUIRES_OK(context, context->GetAttr("num_buckets", &num_buckets_));
    OP_REQUIRES_OK(context, context->GetAttr("feature_column_type", &feature_column_type_));

    static bool printed = false;
    if (!printed) {
      printf("******** CategoricalColumnWithHashBucketOp ********\n");
      printed = true;
    }
  }

  ~CategoricalColumnWithHashBucketOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Input
    const Tensor& input_tensor = context->input(0);
    const Tensor& weight_tensor = context->input(1);

    int num_input_dim = input_tensor.dims();
    Tensor *dense_shape_tensor = NULL;
    TensorShape dense_shape_shape({num_input_dim});
    OP_REQUIRES_OK(context, context->allocate_output(2, dense_shape_shape, &dense_shape_tensor));
    int64 *dense_shape = (int64 *)dense_shape_tensor->tensor_data().data();

    // Collect dense shape
    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
      *dense_shape++ = input_tensor.dim_size(i);
    }

    // Get the number of non-empty values
    int non_empty_input = 0;
    std::vector<bool> not_empty(input_size);
    Tid *input = (Tid *)input_tensor.tensor_data().data();
    Tweight *weight = (Tweight *)weight_tensor.tensor_data().data();
    for (int i = 0; i < input_size; i++) {
      if (*input++ == *weight) {
        not_empty[i] = false;
      } else {
        not_empty[i] = true;
        non_empty_input++;
      }
    }

    Tensor *indices_tensor = NULL;
    TensorShape indices_shape({non_empty_input, num_input_dim});
    OP_REQUIRES_OK(context, context->allocate_output(1, indices_shape, &indices_tensor));

    std::vector<string> values(non_empty_input);
    input = (Tid *)input_tensor.tensor_data().data();
    int64 *indices = (int64 *)indices_tensor->tensor_data().data();

    // Copy non-empty values and their indice
    if (num_input_dim == 2) {
      assign_values_2dims(input_tensor.dim_size(0),
                          input_tensor.dim_size(1),
                          not_empty, input, values, indices);
    } else if (num_input_dim == 3) {
      assign_values_3dims(input_tensor.dim_size(0),
                          input_tensor.dim_size(1),
                          input_tensor.dim_size(2),
                          not_empty, input, values, indices);
    } else {
      assign_values_ndims(input_tensor, num_input_dim,
                          not_empty, input, values, indices);
    }

    if ( feature_column_type_ == "CategoricalColumnWithHashBucket" ) {
      Tensor *hash_tensor = nullptr;
      TensorShape hash_shape({non_empty_input});
      OP_REQUIRES_OK(context, context->allocate_output(0, hash_shape, &hash_tensor));
      auto hash = hash_tensor->flat<int64>();

      for (int i = 0; i < values.size(); ++i) {
        const uint64 input_hash = Fingerprint64(values[i]);
        const uint64 bucket_id = input_hash % num_buckets_;
        hash(i) = static_cast<int64>(bucket_id);
      }
    }
  }

private:
  inline void assign_values_2dims(int rows, int cols,
                                  const std::vector<bool>& not_empty,
                                  Tid* input, std::vector<string>& values, int64* indices) {
    int index = 0;
    for (int i = 0; i < rows; i++) {
      for (int j = 0; j < cols; j++) {
        if (not_empty[index]) {
          values[i] = input[index];
          *indices++ = i;
          *indices++ = j;
        }
        index += 1;
      }
    } // end for i
  }

  inline void assign_values_3dims(int dim1, int dim2, int dim3,
                                  const std::vector<bool>& not_empty,
                                  Tid* input, std::vector<string>& values, int64* indices) {
    int index = 0;
    for (int i = 0; i < dim1; i++) {
      for (int j = 0; j < dim2; j++) {
        for (int k = 0; k < dim3; k++) {
          if (not_empty[index]) {
            values[i] = input[index];
            *indices++ = i;
            *indices++ = j;
            *indices++ = k;
          }
          index += 1;
        } // end for k
      } // end for j
    } // end for i
  }

  inline void assign_values_ndims(const Tensor& input_tensor, int num_input_dim,
                                  const std::vector<bool>& not_empty,
                                  Tid* input, std::vector<string>& values, int64* indices) {
    int64 dim_size[num_input_dim];
    int64 cur_indice[num_input_dim];
    int64 input_size = 1;
    for (int k = 0; k < num_input_dim; ++k) {
      dim_size[k] = input_tensor.dim_size(k);
      input_size *= input_tensor.dim_size(k);
      cur_indice[k] = 0;
    }

    for (int i = 0; i < input_size; ++i) {
      if (not_empty[i]) {
        values[i] = input[i];
        // Copy indice value
        for (int k = 0; k < num_input_dim; ++k) {
          indices[k] = cur_indice[k];
        }
        indices += num_input_dim;
      }
      // increase the indice
      int increment_dim = num_input_dim - 1;
      while (increment_dim >= 0) {
        if (cur_indice[increment_dim] + 1 < dim_size[increment_dim]) {
          cur_indice[increment_dim] += 1;
          break;
        }
        cur_indice[increment_dim] = 0;
        increment_dim -= 1;
      } // end while
   } // end for
  }

private:
  int64 num_buckets_;
  std::string node_name;
  std::string feature_column_type_;
};



REGISTER_KERNEL_BUILDER(Name("CategoricalColumnWithHashBucket")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<tstring>("Tid")
                            .TypeConstraint<tstring>("Tweight"),
                        CategoricalColumnWithHashBucketOp<tstring, tstring>);

REGISTER_KERNEL_BUILDER(Name("CategoricalColumnWithHashBucket")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tid")
                            .TypeConstraint<int64>("Tweight"),
                        CategoricalColumnWithHashBucketOp<int64, int64>);
