/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/grappler/optimizers/bucketize_optimizer.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"
#include <iostream>

using namespace std;


namespace tensorflow {
namespace grappler {

  namespace bucketize_optimizer{
//infomation help to identify tensor is output to which node and the corresponding input number
struct outNodeInfo{
   string node_name;
   int32_t input_number;
};



DataType GetInputType(GraphProperties& properties, const NodeDef* node_view, int i_th) {

    const string &name = node_view->name();
    const auto& input = properties.GetInputProperties(name);

    if (i_th + 1 > input.size()) {
        LOG(WARNING) << "Cannot get data type for input " << i_th << " of node " << name;
        return DT_INT64;
    }

    const DataType type = input[i_th].dtype();
    return type;
}



string get_node_by_tensor(string tensor_name) {
    auto position = tensor_name.find(":");
    if ( position != string::npos )
        tensor_name.erase(tensor_name.begin() + position, tensor_name.end());
    if ( tensor_name[0] == '^' )
        tensor_name.erase(tensor_name.begin());

    return tensor_name;
}



NodeDef* find_input_node_by_op(std::unordered_map<string, NodeDef*>& node_mapping, NodeDef* node, string op)
{
    int input_num = node->input_size();
    for (int i = 0; i < input_num; i++)
    {
       string input_node_name = get_node_by_tensor(node->input(i));
       
       NodeDef* node = node_mapping[input_node_name];

       if (node->op() == op)
       {
          return (node);
       }

    }
    
    return (nullptr);
}

NodeDef* match_bucketize_node(std::unordered_map<string, NodeDef*>& node_mapping, string reshape_name)
{
    NodeDef* bucketize_node = nullptr;
    
    //bucketize -> cast -> onehot -> reshape
    
    NodeDef* reshape_node = node_mapping[reshape_name];
    
    //find onehot
    NodeDef* onehot = find_input_node_by_op(node_mapping, reshape_node, "OneHot"); 
    
    if (onehot == nullptr)   
    {
      return nullptr;    
    }
    //find cast
    NodeDef* cast = find_input_node_by_op(node_mapping, onehot, "Cast"); 
    
    if (cast == nullptr)
    {
      return nullptr;
    }

    bucketize_node = find_input_node_by_op(node_mapping, cast, "Bucketize"); 
    
    if (bucketize_node == nullptr)
    {
      return nullptr;
    }

    return (bucketize_node);
}

}

Status BucketizeOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
    *optimized_graph = item.graph;
    bool enableBucketizeFuse = true;

    const char* env_p = std::getenv("BUCKETIZE_FUSE");
    if (env_p != NULL && env_p[0] == '0') {
        LOG(INFO) << "Ignore the optimization as EC_FUSE=0";
        //return Status::OK();
	      enableBucketizeFuse = false;
    }
   
    using namespace bucketize_optimizer;

    if (!enableBucketizeFuse)
    {
	//nothing to do!!    
        return Status::OK();
    }
    
    // collect the common information
    std::unordered_map<string, std::vector<string>> node_outputs;
    std::unordered_map<string, NodeDef*> node_mapping;
    
    unordered_map<string, vector<outNodeInfo*>> tensor_to_node;
   

    for (auto& node : *optimized_graph->mutable_node()) {
        string node_name = node.name();
        node_mapping.insert(std::make_pair(node_name, &node));
        
	      int input_num = node.input_size();
        if ( input_num == 0 ) continue;

        for ( int i = 0; i < input_num; i ++ ) {
            //get node name
    	    string prenode_name = get_node_by_tensor(node.input(i));
            //get input tensor name, and map to output node
    	    string input_tensor = node.input(i);
            if (input_tensor[0] != '^')
    	    {
                    			
    	        auto out_info = new outNodeInfo;
                out_info->node_name = node_name;
    	        out_info->input_number = i;

    	        //ignore control input
    	        if (tensor_to_node.count(input_tensor) > 0)
    	        {
                    tensor_to_node[input_tensor].push_back(out_info);   
    	        }
    	        else
    	        {
                    vector<outNodeInfo*> tmp_vector = {out_info};
    	    	    tensor_to_node.insert(make_pair(input_tensor, tmp_vector));
    	        }
    	    }	


            if( node_outputs.count(prenode_name) > 0 ) {
               node_outputs[prenode_name].push_back(node_name);
            } else {
               std::vector<string> tmp_vector = {node_name};
               node_outputs.insert(std::make_pair(prenode_name, tmp_vector));
            }
        }
    }

    if (enableBucketizeFuse)
    {
        printf("\n [Begin] - Bucketize Optimizer.\n");
        std::vector<string> concat_nodes;
        string node_device = "";
        for (auto& node : *optimized_graph->mutable_node()) {
            string node_op = node.op();
            // Record all nodes with 'ConcatV2' type
            if ( node_op == "ConcatV2" ) {
                string node_name = node.name();
                concat_nodes.push_back(node_name);
                node_device = node.device();
            }
        }
        if ( concat_nodes.size() == 0 ) {
            printf(" [Over] - concat node is not found.\n");
            return Status::OK();
        }

        printf("    [Running] - Discovered %d subgraphs to be merged.\n", concat_nodes.size());

        // fuse
        GraphProperties properties(item);
        properties.InferStatically(false);

        int optimizer_idx = 0;
        for ( int i = 0; i < concat_nodes.size(); i ++ ) {
            
            int input_num = node_mapping[concat_nodes[i]]->input_size();  
            
            std::vector<NodeDef* > bucketize_nodes;
            std::vector<NodeDef* > const_node;

            for (int j = 0; j < input_num; j++)
            {
              
              string input_node_name = get_node_by_tensor(node_mapping[concat_nodes[i]]->input(j));
              
              NodeDef* input_node = node_mapping[input_node_name];

              if (input_node == nullptr || (input_node->op() != "Const" &&  input_node->op() != "Reshape"))
              {
                printf("break Concat node due to input_node null or input_node OP mismatch. \n");
                break;
              }
              
              if ( input_node->op() == "Reshape" )
              { 
                  NodeDef* bucketize_node = match_bucketize_node(node_mapping, input_node_name);
                  bucketize_nodes.push_back(bucketize_node);   
              }
              else if (input_node->op() == "Const")
              {
                  const_node.push_back(input_node);
              }
            }
             
            if ((bucketize_nodes.size() + const_node.size()) != input_num)
            {
                printf("concat node input mismatch, move to next concat node. \n");
                continue;
            }
             

            // Create fuse bucketize node
            string fuse_node_name = concat_nodes[i] + "_fused";
            
            NodeDef* fuse_node = optimized_graph->add_node();
            
            fuse_node->set_op("FuseBucketizeConcat");

	          fuse_node->set_name(fuse_node_name);

            fuse_node->set_device(node_device);
            
            DataType input_type;
            string tensor_name;
            
            (*fuse_node->mutable_attr())["boundaries_size"].mutable_list()->Clear();
            (*fuse_node->mutable_attr())["boundaries"].mutable_list()->Clear();

            for (int j = 0 ; j < bucketize_nodes.size(); j++)
            {
#if 1
                input_type = GetInputType(properties, node_mapping[concat_nodes[i]] , j);
                //printf("concatV2 input %d type %d \n", j, input_type);
               
                tensor_name = bucketize_nodes[j]->input(0); 
                //printf("bucketize input name: %s \n", tensor_name.c_str());

#endif
                fuse_node->add_input(tensor_name);
                
                //copy boundaries 
                auto boundaries_list = (*bucketize_nodes[j]->mutable_attr())["boundaries"].list();
                
                auto size = boundaries_list.f_size(); 
                
                //printf("boundaries size: %d \n", size);
                 
                for (int k = 0; k < size; k++)
                {
                  (*fuse_node->mutable_attr())["boundaries"].mutable_list()->add_f(boundaries_list.f(k));     
                }
                
                (*fuse_node->mutable_attr())["boundaries_size"].mutable_list()->add_i(size);

            }

            (*fuse_node->mutable_attr())["T"].set_type(input_type);
            (*fuse_node->mutable_attr())["N"].set_i(bucketize_nodes.size()); 
            (*fuse_node->mutable_attr())["use_avx"].set_b(false);                      
            
            if( node_outputs.count(concat_nodes[i]) == 0 ) {
                printf(" [WARNING] - %s has no output.", concat_nodes[i].c_str());
                return Status::OK();
            }
            
            // Replace the input node to the new node
            auto concat_name = concat_nodes[i];
            auto nodes = node_outputs[concat_name];
            for ( int i = 0; i < nodes.size(); i ++ ) {
                string stop_nodename = nodes.at(i);
                for (auto& node : *optimized_graph->mutable_node()) {
                    if ( node.name() != stop_nodename ) continue;

                    for ( int i = 0; i < node.input_size(); i++ ) {
                        
                        if ( node.input(i) == concat_name ) {

                            printf("node %s input %d replaced to %s \n", node.name().c_str(), i, fuse_node_name.c_str());
                          
                            node.set_input(i, fuse_node_name);

                            break;
                        }
                    }

                    break;
                }
            }
             
      
          optimizer_idx++;

          optimizer_run_ = true;
       } 
       
        printf(" [Done] - BucketizeOptimizer %d concat node .\n", optimizer_idx );
	  }

    
    return Status::OK();
}

void BucketizeOptimizer::Feedback(Cluster* /*cluster*/,
                                const GrapplerItem& /*item*/,
                                const GraphDef& /*optimized_graph*/,
                                double /*result*/) {
    // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // namespace tensorflow
