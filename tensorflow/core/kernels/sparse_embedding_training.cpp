#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/resource_variable_ops.h"
#include "tensorflow/core/kernels/variable_ops.h"

using namespace tensorflow;
using namespace std;

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
static void sparse_gather(Tid *input, int64 input_size, Tshape *indice, int indice_dim, Tshape *shape, int rows, int cols, float *embedding_table, float *output, int embedding_size, bool is_mean, bool *p_output6, uint32_t* row_idx) {
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
    
    assert (row < rows);
    assert (row_values[row] <= 1);
    //std::cout<<"sparse_gather.... row:" <<row<<"  input_idx:"<< i <<std::endl;
    row_idx[row] = i;     

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
      if (p_output6 != nullptr)
      {
          for (uint32_t j = 0; j < embedding_size; j++)
          {
              p_output6[i*embedding_size + j] = true;
          }
      }
    } else if (is_mean && row_values[i] > 1) {
      float factor = 1.0f / row_values[i];
      myscale(&output[i * embedding_size], factor, embedding_size);
    }
  }

  delete[] row_values;
}

REGISTER_OP("SparseEmbeddingWithShapeTraining")
    .Input("weight: Tweight")
    .Input("input: Tid")
    .Input("dense_shape: Tshape")
    .Input("indice: Tshape")
    .Output("embedded: float")
    .Output("unique0: Tid")
    .Output("uinque1: int32")
    .Output("resource_gather: float32")
    .Output("cast: int32")
    .Output("zeros_like: float32")
    .Output("tile: bool")
  /*  .Output("shape: int32") */
    .Attr("Combiner: int")
    .Attr("Tid: {int64, int32}")
    .Attr("Tshape: {int64, int32}")
    .Attr("Tweight: {float, resource}");
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


template<typename Tid, typename Tshape>
class SparseEmbeddingWithShapeTrainingOp : public OpKernel {
public:
  explicit SparseEmbeddingWithShapeTrainingOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("Combiner", &combiner));
    //OP_REQUIRES_OK(context, context->GetAttr("Dims", &dims));
    node_name = context->def().name();

    static bool printed = false;
    if (!printed) {
      printf("******** SparseEmbeddingWithShapeOp ********\n");
      printed = true;
    }
    
  }

  ~SparseEmbeddingWithShapeTrainingOp() {
  }

  void Compute(OpKernelContext* context) override {
    // Grab the weight
    float *weight;
    const Tensor* weight_tensor = &context->input(0);
    if (weight_tensor->dtype() == DT_RESOURCE) {
      Var* variable;
      OP_REQUIRES_OK(context,
                     LookupResource(context, HandleFromInput(context, 0), &variable));
      core::ScopedUnref s(variable);
      weight_tensor = variable->tensor();
      OP_REQUIRES(context, weight_tensor->dtype() == DT_FLOAT,
                  errors::InvalidArgument("Expect float weight in ", node_name));
    }
    weight = (float *)weight_tensor->tensor_data().data();
    
    // Input id
    const Tensor& input_tensor = context->input(1);
    Tid *input = (Tid *)input_tensor.tensor_data().data();
    
    const Tensor& shape_tensor = context->input(2);
    Tshape *shape = (Tshape *)shape_tensor.tensor_data().data();

    // To check the input
    OP_REQUIRES(context, (shape_tensor.dims() == 1),
                errors::InvalidArgument("Shape tensor is not valid (dims != 1)"));
    OP_REQUIRES(context, (shape_tensor.dim_size(0) >= 2),
                errors::InvalidArgument("Shape tensor is not valid (dim_size(0) >= 2)"));



    int64 input_size = 1;
    for (int i = 0; i < input_tensor.dims(); ++i) {
      input_size *= input_tensor.dim_size(i);
    }
    
    int input_dims = shape_tensor.dim_size(0);
    int cols = shape[input_dims - 1];
    int batch_size = 1;
    for (int i = 0; i < input_dims - 1; ++i) {
      batch_size *= shape[i];
    }
    int embedding_size = weight_tensor->dim_size(1);
    bool is_mean = (combiner == 1);

    //OP_REQUIRES(context, (input_size == (batch_size * cols)), errors::InvalidArgument("input is not dense tensor"));

    //if (!(input_size == (batch_size * cols)))
    //{ 
    //    std::cout<<"\n sparseSegmentMeanWithShape input_size:"<<input_size<<" batch_size:"<<batch_size<<" cols:"<<cols<<std::endl;
    //}

    // Create an output tensor
    Tensor* output_tensor = NULL;
    TensorShape output_shape({batch_size, embedding_size});
    /* Testing shows following impl. is not correct
    if (input_dims == 2) {
      output_shape = TensorShape({batch_size, embedding_size});
    } else {
      std::vector<int> tmp_sizes;
      for (int i = 0; i < input_dims - 1; ++i) {
        tmp_sizes.push_back(shape[i]);
      }
      tmp_sizes.push_back(embedding_size);
      OP_REQUIRES_OK(context, TensorShapeUtils::MakeShape(tmp_sizes.data(), tmp_sizes.size(), &output_shape));
    }*/
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape,
                                                     &output_tensor));
    float *output = (float *)output_tensor->tensor_data().data();

    
    m_p_output3 = nullptr;
    m_p_output6 = nullptr;
    

    uint32_t row_index[batch_size];    
 
    //std::cout<<"step1 !"<<std::endl;
  
   
    {
         //output6 need to be initialized firstly, tile output, bool to indicate any default value filled in sparse_gather
         Tensor* output6 = NULL;
         TensorShape output_shape6({batch_size, embedding_size});
         OP_REQUIRES_OK(context, context->allocate_output(6, output_shape6,
                	                                     &output6));
             
         m_p_output6 = (bool *)output6->tensor_data().data();
         
         memset(m_p_output6, 0, batch_size * embedding_size * sizeof(bool));
     }
     
    //std::cout<<"step2 !"<<std::endl;
  
    if (input_size == batch_size * cols) { // input id is dense
      sparse_gather(input, batch_size, cols, weight, output, embedding_size, is_mean);
      
      assert(cols == 1);
      
      //for dense tensor
      for (uint32_t i = 0; i < batch_size; i++)
      {
          row_index[i] = i; 
      }      

    } else { // input id is sparse
      const Tensor& indice_tensor = context->input(3);
      OP_REQUIRES(context, (indice_tensor.dims() == 2),
                  errors::InvalidArgument("Indice tensor is not as expected (dims != 2)"));
      OP_REQUIRES(context, (indice_tensor.dim_size(0) == input_size),
                  errors::InvalidArgument("Indice tensor is not as expected (dim_size(0) != batch_size)"));
      Tshape *indice = (Tshape *)indice_tensor.tensor_data().data();
      int indice_dim = indice_tensor.dim_size(1);
      sparse_gather(input, input_size, indice, indice_dim, shape, batch_size, cols, weight, output, embedding_size, is_mean, m_p_output6, row_index);
 
    }

    { 
         //uqique input, unique:0 for out 1, unique:1 for out 2
         uint32_t idx_in_unique_vec[batch_size]; //record the index to value in unique array
         m_unique_set.clear();
         m_unique_val.clear();
         m_unique_idx.clear();         
        
         //std::cout<<"step3 !"<<std::endl;

         Tid tmp_input[batch_size];
         for (uint32_t i = 0; i < batch_size; i++)         
         {
             if (m_p_output6[i*embedding_size] == true)
             {
                tmp_input[i] = 0;
             }
             else
             {
                //std::cout << "step3....row:" <<i<<" input idx:"<<row_index[i]<<std::endl;
                tmp_input[i] = input[row_index[i]];
             }
         }

         //std::cout<<"step4 !"<<std::endl;

         for (uint32_t i = 0; i < batch_size; i++)
         {
             if (m_unique_set.insert(tmp_input[i]).second)
             {
                 idx_in_unique_vec[i] = m_unique_val.size(); 	
                 m_unique_val.push_back(tmp_input[i]);
                 m_unique_idx.push_back(i);
             }
             else
             {
                 //duplicate element
                 for (uint32_t j = 0; j < m_unique_val.size(); j++)
                 {
                     if (m_unique_val[j] == tmp_input[i])
             	    {
             	        idx_in_unique_vec[i] = j; 
             	        break;
             	    }
                 }
             }
         }

    
         //std::cout<<"step5 !"<<std::endl;

         Tensor* output1 = NULL;
         TensorShape output_shape1({m_unique_val.size()});
         OP_REQUIRES_OK(context, context->allocate_output(1, output_shape1,
                                                             &output1));
         Tid *p_output1 = (Tid *)output1->tensor_data().data(); 
         
         Tensor* output2 = NULL;
         TensorShape output_shape2({batch_size});
         OP_REQUIRES_OK(context, context->allocate_output(2, output_shape2,
                                                             &output2));
         int32_t *p_output2 = (int32_t *)output2->tensor_data().data();
            
         //resource gather output, with shape of size(unique:0) * embedding_size
         Tensor* output3 = NULL;
         TensorShape output_shape3({m_unique_val.size(), embedding_size });
         OP_REQUIRES_OK(context, context->allocate_output(3, output_shape3,
                                                          &output3));
         m_p_output3 = (float *)output3->tensor_data().data();
          
         //cast output, an array from 0 to batch size
         Tensor* output4 = NULL;
         TensorShape output_shape4({batch_size});
         OP_REQUIRES_OK(context, context->allocate_output(4, output_shape4,
                                                             &output4));
         int32_t *p_output4 = (int32_t *)output4->tensor_data().data();
         
         //zeros_like output, with the same shape of output 0
         Tensor* output5 = NULL;
         TensorShape output_shape5({batch_size, embedding_size});
         OP_REQUIRES_OK(context, context->allocate_output(5, output_shape5,
                		                             &output5));
             
         float *p_output5 = (float *)output5->tensor_data().data();
 
         ////tile output, bool to indicate any default value filled in sparse_gather
         //Tensor* output6 = NULL;
         //TensorShape output_shape6({batch_size, embedding_size});
         //OP_REQUIRES_OK(context, context->allocate_output(6, output_shape6,
         //       	                                     &output6));
         //    
         //m_p_output6 = (bool *)output6->tensor_data().data();
        

         for (uint32_t i = 0; i < m_unique_val.size(); i++)
         {
              p_output1[i] = m_unique_val[i];
         }

         //std::cout << "\n SparseSegmentSum output 2:"<<std::endl;
         for (uint32_t i = 0; i < batch_size; i++)
         {
             p_output2[i] = idx_in_unique_vec[i];
	  //   cout<<" index: "<< i <<"values: " <<p_output2[i]; 
         }
        
	 //cout<<endl;

         for (uint32_t i = 0; i < batch_size; i++)
         {
              p_output4[i] = i;
         }

          memset(p_output5, 0, batch_size * embedding_size * sizeof(float));
          
    	    
       
          //fill output3 from output
          for (uint32_t i = 0; i < m_unique_idx.size(); i++)
          {   
              for (uint32_t j = 0; j < embedding_size; j++)
              {
                  m_p_output3[i*embedding_size + j] = output[m_unique_idx[i]*embedding_size + j];
              }
          }
        
    }

  }

private:
  // 0=SUM, 1=MEAN
  int combiner;
  std::string node_name;
  
  set<Tid> m_unique_set; 
  vector<Tid> m_unique_val;
  vector<uint32_t> m_unique_idx; //record the first unique value index
  float* m_p_output3;
  bool* m_p_output6;
};



REGISTER_KERNEL_BUILDER(Name("SparseEmbeddingWithShapeTraining")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int32>("Tid")
                            .TypeConstraint<int64>("Tshape"),
                        SparseEmbeddingWithShapeTrainingOp<int32, int64>);
REGISTER_KERNEL_BUILDER(Name("SparseEmbeddingWithShapeTraining")
                            .Device(DEVICE_CPU)
                            .TypeConstraint<int64>("Tid")
                            .TypeConstraint<int64>("Tshape"),
                        SparseEmbeddingWithShapeTrainingOp<int64, int64>);
