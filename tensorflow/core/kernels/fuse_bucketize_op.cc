#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/register_types.h"
#include <iostream>
#include <immintrin.h>
#include <chrono>


using namespace std;


namespace tensorflow { 

using CPUDevice = Eigen::ThreadPoolDevice;


REGISTER_OP("FuseBucketize")
    .Input("input: T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("boundaries: list(float)")
    .Attr("use_avx: bool = false");

REGISTER_OP("FuseBucketizeConcat")
    .Input("input: N * T")
    .Output("output: T")
    .Attr("T: type")
    .Attr("N: int >= 1")
    .Attr("boundaries_size: list(int)") // N size
    .Attr("boundaries: list(float)")  //N boundaries
    .Attr("use_avx: bool = false");




template <typename Device, typename T>
class FuseBucketizeOp : public OpKernel {
  public:
    FuseBucketizeOp(OpKernelConstruction* c) : OpKernel(c){
        
        OP_REQUIRES_OK(c, c->GetAttr("boundaries", &boundaries_));
        OP_REQUIRES(c, std::is_sorted(boundaries_.begin(), boundaries_.end()),
                errors::InvalidArgument("Expected sorted boundaries"));
      
        if (c->HasAttr("use_avx")){
          OP_REQUIRES_OK(c, c->GetAttr("use_avx", &m_use_avx)); 
    
        }   

  } 

  void Compute(OpKernelContext *c) override {
    const Tensor& input_tensor = c->input(0);
    const auto input = input_tensor.flat<T>();

    TensorShape input_shape = input_tensor.shape();

    OP_REQUIRES(
        c, input_shape.dims() == 2 && input_shape.dim_size(1) == 1,
        errors::InvalidArgument("Expected input_dims to be 2 but get dims: ",
                                input_shape.dims(), "dim 1 size: ", input_shape.dim_size(1)));
     

    const int N = input.size();
    int32_t  indices[N];
    
    for (int i = 0; i < N; i ++)
    {
      auto first_bigger_it = std::upper_bound(
          boundaries_.begin(), boundaries_.end(), input(i));
      indices[i] = first_bigger_it - boundaries_.begin();
    }
    
    int dim2 = boundaries_.size() + 1;

    TensorShape output_shape({N, dim2});
    
    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, output_shape, &output)); 

    auto out_mat = output->matrix<T>();

    out_mat.setZero();

    for (int i = 0; i < N; i++)
    {
      out_mat(i, indices[i]) = 1;
    }
    
    

#if 0
/           auto start = chrono::steady_clock::now();

         
//           auto end = chrono::steady_clock::now();

//           cout << "avx time: " << chrono::duration_cast<chrono::microseconds>(end - start).count() << " us" << endl; 

       
#endif
  }

private:
  bool m_use_avx = false;
  std::vector<float> boundaries_; 
};

template <typename Device, typename T>
class FuseBucketizeConcatOp : public OpKernel {
  public:
    FuseBucketizeConcatOp(OpKernelConstruction* c) : OpKernel(c){
        
        OP_REQUIRES_OK(c, c->GetAttr("boundaries", &boundaries_));
        OP_REQUIRES_OK(c, c->GetAttr("boundaries_size", &boundaries_size_));
        
        
        OP_REQUIRES(c, boundaries_size_.size() > 0, errors::InvalidArgument("boundaries size must set"));
        
        m_concat_N = boundaries_size_.size();

        std::vector<float>::iterator  start_index = boundaries_.begin();
        std::vector<float>::iterator  end_index;
         
        boundaries_array_.resize(m_concat_N);
        
        boundaries_start_index_.resize(m_concat_N);

        for (int i = 0; i < m_concat_N; i++)
        {
            end_index = start_index + boundaries_size_[i];
            
            if (i > 0)
            {
                //start index for each boundary one-hoted
                boundaries_start_index_[i] = boundaries_start_index_[i - 1] +  boundaries_size_[i - 1] + 1;
            }
            
            boundaries_array_[i].assign(start_index, end_index);
                        
            //std::cout << *(boundaries_array_[i].begin()) << "  " << *(boundaries_array_[i].end() - 1 )  << std::endl;  
            
            //std::cout <<"boundaries: " << i << "size: " << boundaries_size_[i] <<" start_index :"<<  boundaries_start_index_[i] << " "<< boundaries_array_[i].size() << std::endl;
           
            OP_REQUIRES(c, std::is_sorted(boundaries_array_[i].begin(), boundaries_array_[i].end()), errors::InvalidArgument("Expected sorted boundaries"));
            
            start_index = end_index;

            m_output_dim1 += (boundaries_size_[i] + 1); 
        }
        
        if (c->HasAttr("use_avx")){
          OP_REQUIRES_OK(c, c->GetAttr("use_avx", &m_use_avx)); 
    
        }   

  } 

  void Compute(OpKernelContext *c) override {
    
    OpInputList input;

    OP_REQUIRES_OK(c, c->input_list("input", &input));
    
    const int concat_N = input.size();
    
    const Tensor& input0 = input[0];
    
    const TensorShape &input_shape = input0.shape();
    
    const int output_dim0 = input_shape.dim_size(0);
 
    OP_REQUIRES(
        c, m_concat_N == concat_N,
        errors::InvalidArgument("input tensor num must match boundaries num, boundaries num: ",
                                m_concat_N, " input tensor num : ", concat_N ));
    
    TensorShape out_shape({output_dim0, m_output_dim1});
    
    int32_t indices[output_dim0][m_concat_N];
    
    for (int i = 0; i < concat_N; i++)
    {
      const auto input_i = input[i].flat<T>();
      for (int j = 0; j < output_dim0; j++)
      {
          auto first_bigger_it = std::upper_bound(
          boundaries_array_[i].begin(), boundaries_array_[i].end(), input_i(j));
          indices[j][i] = first_bigger_it - boundaries_array_[i].begin() + boundaries_start_index_[i];
      }
    }


    Tensor* output;
    OP_REQUIRES_OK(c, c->allocate_output(0, out_shape, &output)); 

    auto out_mat = output->matrix<T>();

    out_mat.setZero();
     
    for (int i = 0; i < output_dim0; i++)
    {  
        for (int j = 0; j < concat_N; j++)
        {
            out_mat(i, indices[i][j]) = 1;
        }
    }
  }

private:
  std::vector<float> boundaries_;
  std::vector<int> boundaries_size_;
  std::vector<int> boundaries_start_index_;
  std::vector< std::vector<float> > boundaries_array_;
  int m_concat_N = 0;
  int m_output_dim1 = 0;
  bool m_use_avx = false;
};

REGISTER_KERNEL_BUILDER(Name("FuseBucketize")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<float>("T"),
                        FuseBucketizeOp<CPUDevice, float>);

  
 REGISTER_KERNEL_BUILDER(Name("FuseBucketizeConcat")
                        .Device(DEVICE_CPU)
                        .TypeConstraint<float>("T"),
                        FuseBucketizeConcatOp<CPUDevice, float>);



}
