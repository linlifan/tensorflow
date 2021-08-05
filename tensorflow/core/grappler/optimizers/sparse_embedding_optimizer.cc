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

#include "tensorflow/core/grappler/optimizers/sparse_embedding_optimizer.h"

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

// Seems cannot get shape during the optimization, carefully to call it
int GetOutputDims(GraphProperties& properties, const string& node_name) {
  printf("Try to get shape\n");
  const std::vector<OpInfo::TensorProperties>& prop_list =
      properties.GetOutputProperties(node_name);
  const OpInfo::TensorProperties& props = prop_list[0];
  TensorShape shape(props.shape());
  printf("Dims is %d\n", shape.dims());
  return shape.dims();
}

string get_node_by_tensor(string tensor_name) {
    auto position = tensor_name.find(":");
    if ( position != string::npos )
        tensor_name.erase(tensor_name.begin() + position, tensor_name.end());
    if ( tensor_name[0] == '^' )
        tensor_name.erase(tensor_name.begin());

    return tensor_name;
}

NodeDef* skip_identity(std::unordered_map<string, NodeDef*>& node_mapping, string node_name) {
    NodeDef* node = node_mapping[node_name];
    while ( node->op() == "Identity" ) {
        node_name = get_node_by_tensor(node->input(0));
        node = node_mapping[node_name];
    }
    return node;
}

NodeDef* find_output_node(std::unordered_map<string, std::vector<string>>& node_outputs,
                    std::unordered_map<string, NodeDef*>& node_mapping,
                    string node_name, std::vector<string>& target_ops) {
    if( node_outputs.count(node_name) == 0 ) return NULL;

    auto nodes = node_outputs[node_name];
    for ( int i = 0; i < nodes.size(); i ++ ) {
        string outnode_name = nodes.at(i);
        NodeDef* tmp_outnode = node_mapping[outnode_name];
        if ( find(target_ops.begin(), target_ops.end(), tmp_outnode->op()) != target_ops.end() )
            return tmp_outnode;
    }
    return NULL;
}

bool match_unsorted_segment_sum_pattern(uint32_t pattern, 
                                       GraphProperties& properties,
				       NodeDef* root_node, 
				       unordered_map<string, NodeDef*>& node_mapping, 
				       vector<string> & input_tensors, 
				       vector<DataType>& input_types) 
{

    if (pattern == 0 || pattern == 2)
    {
        string root_input_node0 = get_node_by_tensor(root_node->input(0));
        NodeDef* gather_node = node_mapping[root_input_node0]; //skip_identity(node_mapping, root_input_node0);
        
       
	if (gather_node == nullptr || gather_node->op() != "GatherV2") {
            printf(" [Over] - Cannot find GatherV2 for %s.\n", root_node->name().c_str());
            return false;
        }

        string root_input_node1 = get_node_by_tensor(root_node->input(1));
        NodeDef* unique_node = node_mapping[root_input_node1]; 
        
	if (unique_node == nullptr || unique_node->op() != "Unique" &&  unique_node->op() != "SparseEmbeddingWithShape") {
            printf(" [Over] - Cannot find Unique or SparseEmbeddingWithShape for %s.\n", root_node->name().c_str());
            return false;
        }

        string root_input_node2 = get_node_by_tensor(root_node->input(2));
        NodeDef* strided_slice_node = node_mapping[root_input_node2]; 

	if (strided_slice_node != nullptr && strided_slice_node->op() != "StridedSlice") {
            printf(" [Over] - Cannot find StridedSlice for %s.\n", root_node->name().c_str());
            return false;
        }
        
        //check stridedSlice input with Shape
        string strided_slice_input_node0 = get_node_by_tensor(strided_slice_node->input(0));
        NodeDef* shape_node = node_mapping[strided_slice_input_node0]; 
        
	
	if (shape_node != nullptr && shape_node->op() != "Shape") {
            printf(" [Over] - Cannot find Shape for %s.\n", root_input_node2.c_str());
            return false;
        }


        //first input is  GatherV2 input 0
        input_tensors.push_back(gather_node->input(0));
        input_types.push_back(GetInputType(properties, gather_node, 0));
        //second input is root input 1
        input_tensors.push_back(root_node->input(1));
        input_types.push_back(GetInputType(properties, root_node, 1));     
        //other inputs , not used by now
        input_tensors.push_back(gather_node->input(1));
        input_types.push_back(GetInputType(properties, gather_node, 1));     
        
        input_tensors.push_back(gather_node->input(2));
        input_types.push_back(GetInputType(properties, gather_node, 1));     
         
        input_tensors.push_back(shape_node->input(0));
        input_types.push_back(GetInputType(properties, shape_node, 0));

        input_tensors.push_back(strided_slice_node->input(1));
        input_types.push_back(GetInputType(properties, strided_slice_node, 1));     

        input_tensors.push_back(strided_slice_node->input(2));
        input_types.push_back(GetInputType(properties, strided_slice_node, 2));

        input_tensors.push_back(strided_slice_node->input(3));
        input_types.push_back(GetInputType(properties, strided_slice_node, 3));
        
        return true;

    }
    else if (pattern == 1)
    {
        string root_input_node0 = get_node_by_tensor(root_node->input(0));
        NodeDef* reshape_node = node_mapping[root_input_node0]; 
        
	assert(reshape_node != nullptr);

	if (reshape_node->op() != "Reshape") {
            printf(" [Over] - Cannot find Reshape for %s.\n", root_node->name().c_str());
            return false;
	}
        
	string root_input_node1 = get_node_by_tensor(root_node->input(1));
        NodeDef* unique_node = node_mapping[root_input_node1]; 
        
	assert(unique_node != nullptr);
	
	if (unique_node->op() != "Unique") {
            printf(" [Over] - Cannot find Unique for %s.\n", root_node->name().c_str());
            return false;
        }

        string root_input_node2 = get_node_by_tensor(root_node->input(2));
        NodeDef* strided_slice_node = node_mapping[root_input_node2]; 
	
	assert(strided_slice_node != nullptr);
        
	if (strided_slice_node->op() != "StridedSlice") {
            printf(" [Over] - Cannot find StridedSlice for %s.\n", root_node->name().c_str());
            return false;
        }
        
        //check stridedSlice input with Shape
        string strided_slice_input_node0 = get_node_by_tensor(strided_slice_node->input(0));
        NodeDef* shape_node = node_mapping[strided_slice_input_node0]; 
        
	assert(shape_node != nullptr);

	if (shape_node->op() != "Shape") {
            printf(" [Over] - Cannot find Shape for %s.\n", root_input_node2.c_str());
            return false;
        }
 
        
	//first input is root input 0
	input_tensors.push_back(root_node->input(0));
	input_types.push_back(GetInputType(properties, root_node, 0));
	//second input is root input 1
	input_tensors.push_back(root_node->input(1));
        input_types.push_back(GetInputType(properties, root_node, 1));     
	//other inputs , not used by now
        input_tensors.push_back(shape_node->input(0));
        input_types.push_back(GetInputType(properties, shape_node, 0));
        
	return true;
    }
    else
    {
        return (false);
    }

}


Status SparseEmbeddingOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
    *optimized_graph = item.graph;
    bool enableEcFuse = true;
    bool enableEcFuseT = true; 

    const char* env_p = std::getenv("EC_FUSE");
    if (env_p != NULL && env_p[0] == '0') {
        LOG(INFO) << "Ignore the optimization as EC_FUSE=0";
        //return Status::OK();
	enableEcFuse = false;
    }
   
     
    env_p = std::getenv("EC_FUSE_T");
    if (env_p != NULL && env_p[0] == '0') {
        LOG(INFO) << "Ignore optimization for training as EC_FUSE_T = 0\n";
	enableEcFuseT = false;
    }
  
    
    if (!enableEcFuse && !enableEcFuseT)
    {
	//nothing to do!!    
        return Status::OK();
    }
    
    // collect the common information
    std::unordered_map<string, std::vector<string>> node_outputs;
    std::unordered_map<string, NodeDef*> node_mapping;
    
    unordered_map<string, vector<outNodeInfo*>> tensor_to_node;
   
    bool is_training_network = false;

    for (auto& node : *optimized_graph->mutable_node()) {
        string node_name = node.name();
        node_mapping.insert(std::make_pair(node_name, &node));
        
	if (!is_training_network)
	{
	    is_training_network = (string::npos != node_name.find("gradient"));	
	}

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

    printf("\n training network: %d \n", is_training_network);  


    if (enableEcFuse)
    {
        printf("\n [Begin] - SparseEmbeddingOptimizer.\n");
        std::vector<string> sparse_segment_nodes;
        string node_device = "";
        for (auto& node : *optimized_graph->mutable_node()) {
            string node_op = node.op();
            // Record all nodes with 'SparseSegmentSum' or 'SparseSegmentMean' type
            if ( node_op == "SparseSegmentSum" || node_op == "SparseSegmentMean" ) {
                string node_name = node.name();
                sparse_segment_nodes.push_back(node_name);
                node_device = node.device();
            }
        }
        if ( sparse_segment_nodes.size() == 0 ) {
            printf(" [Over] - Target node is not found.\n");
            return Status::OK();
        }

        printf("    [Running] - Discovered %d subgraphs to be merged.\n", sparse_segment_nodes.size());

        // fuse
        GraphProperties properties(item);
        properties.InferStatically(false);

        int optimizer_idx = 0;
        for ( int i = 0; i < sparse_segment_nodes.size(); i ++ ) {
            vector<string> additional_output_tensor = {};
	    
	    // Get the embedding node and its weight
            string combiner_input_node = get_node_by_tensor(node_mapping[sparse_segment_nodes[i]]->input(0));
            NodeDef* gather_node = skip_identity(node_mapping, combiner_input_node);
            if ( gather_node->op() != "GatherV2" && gather_node->op() != "ResourceGather" ) {
                printf(" [Over] - Cannot find gather_node for %s.\n", sparse_segment_nodes[i].c_str());
                return Status::OK();
            }
            
           

            string weight_name = gather_node->input(0);
            
            // Get the real input
            // Definition of SparseFillEmptyRows:
            //SparseFillEmptyRows(
            //  const ::tensorflow::Scope & scope,
            //  ::tensorflow::Input indices,
            //  ::tensorflow::Input values,
            //  ::tensorflow::Input dense_shape,
            //  ::tensorflow::Input default_value
            //)
            NodeDef* unique_node = node_mapping[get_node_by_tensor(gather_node->input(1))];
            if ( unique_node->op() != "Unique" ) {
                printf(" [Over] - Unique node is not detected for %s.\n", gather_node->name().c_str());
                return Status::OK();
            }
            
            NodeDef* sparse_fill_node = node_mapping[get_node_by_tensor(unique_node->input(0))];
            if ( sparse_fill_node->op() != "SparseFillEmptyRows" ) {
                printf(" [Over] - Cannot find SparseFillEmptyRows for %s.\n", gather_node->name().c_str());
                return Status::OK();
            }

            NodeDef* value_node = node_mapping[get_node_by_tensor(sparse_fill_node->input(1))];
            if ( value_node->op() != "GatherV2" ) {
                printf(" [Over] - Value not found for SparseFillEmptyRows: %s.\n", sparse_fill_node->name().c_str());
                return Status::OK();
            }

            string input_name = value_node->input(0);

            // Get the dense shape and indice which have value
            string dense_shape_name = get_node_by_tensor(sparse_fill_node->input(2));
            NodeDef* reshape_node = skip_identity(node_mapping, dense_shape_name); // dense_shape comes from SparseReshape
            if ( reshape_node->op() != "SparseReshape" ) {
                printf(" [Over] - Cannot find reshape node for %s.\n", gather_node->name().c_str());
                return Status::OK();
            }

            string indice_tensor = reshape_node->input(0);
            string dense_shape_tensor = reshape_node->input(1);

            // Find out where the embedding ends
            std::vector<string> target_ops = {"ZerosLike"};
            NodeDef* zeros_like_node = find_output_node(node_outputs, node_mapping, sparse_segment_nodes[i], target_ops);
            if ( !zeros_like_node ) {
                printf(" [Over] - Cannot find ZerosLike node for %s.\n", sparse_segment_nodes[i].c_str());
                return Status::OK();
            }
            
            target_ops = {"Select"};
            NodeDef* select_node = find_output_node(node_outputs, node_mapping, zeros_like_node->name(), target_ops);
            if ( !select_node ) {
                printf(" [Over] - Cannot find Select node for %s.\n", zeros_like_node->name().c_str());
                return Status::OK();
            }

            target_ops = {"Reshape"};
            NodeDef* reshape_node1 = find_output_node(node_outputs, node_mapping, select_node->name(), target_ops);
            if ( !reshape_node1 ) {
                printf(" [Over] - Cannot find Reshape node for %s.\n", select_node->name().c_str());
                return Status::OK();
            }
            NodeDef* reshape_node2 = find_output_node(node_outputs, node_mapping, reshape_node1->name(), target_ops);
            if ( reshape_node2 ) // If follows another reshape
                reshape_node1 = reshape_node2;

            string embedding_end_at = reshape_node1->name();
            NodeDef* combiner_node = node_mapping[sparse_segment_nodes[i]];
            
	    if (enableEcFuseT && is_training_network)
	    { 
	        //additional output unique:0, output to 2 grad calculation node 
	        additional_output_tensor.push_back(unique_node->name());
                //additional output unique:1, output to 1 grad calculation node
	        additional_output_tensor.push_back(unique_node->name() + ":1");
                //additional output, output to 1 grad calculation node
	        additional_output_tensor.push_back(gather_node->name());  
 
	        //find cast node for additional output
	        string ssm_input_node2 = get_node_by_tensor(node_mapping[sparse_segment_nodes[i]]->input(2));
                NodeDef* cast_node = skip_identity(node_mapping, ssm_input_node2);
                
	        if (cast_node != nullptr && cast_node->op() == "Cast")
	        {
	            additional_output_tensor.push_back(cast_node->name());
	        }

                additional_output_tensor.push_back(zeros_like_node->name());

	        string select_input = get_node_by_tensor(select_node->input(0));
	        NodeDef* tile_node = skip_identity(node_mapping, select_input);
                if (tile_node != nullptr && tile_node->op() == "Tile")
	        { 
	            additional_output_tensor.push_back(tile_node->name());
	        }

	        additional_output_tensor.push_back(select_node->name());

	    }

            // Create the new node
            string fuse_node_name = reshape_node1->name() + "_fused";
            DataType Tweight_value = GetInputType(properties, gather_node, 0);
            DataType Tshape_value = GetInputType(properties, reshape_node, 1);
            DataType Tid_value = GetInputType(properties, value_node, 0);
            int Combiner_value = 0;
            if ( combiner_node->op() == "SparseSegmentMean" ) Combiner_value = 1;

            NodeDef* fuse_node = optimized_graph->add_node();
            
	    if (enableEcFuseT && is_training_network)
	    {   
               fuse_node->set_op("SparseEmbeddingWithShapeTraining");
	    }
	    else
	    {
               fuse_node->set_op("SparseEmbeddingWithShape");
	    }

	    fuse_node->set_name(fuse_node_name);
            fuse_node->set_device(node_device);
            fuse_node->add_input(weight_name);
            fuse_node->add_input(input_name);
            fuse_node->add_input(dense_shape_tensor);
            fuse_node->add_input(indice_tensor);
            (*fuse_node->mutable_attr())["Tweight"].set_type(Tweight_value);
            (*fuse_node->mutable_attr())["Tshape"].set_type(Tshape_value);
            (*fuse_node->mutable_attr())["Tid"].set_type(Tid_value);
            (*fuse_node->mutable_attr())["Combiner"].set_i(Combiner_value);

            if( node_outputs.count(embedding_end_at) == 0 ) {
                printf(" [Over] - %s has no output.", embedding_end_at.c_str());
                return Status::OK();
            }
            
	    //update node_mapping 
	    //node_mapping.insert(make_pair(fuse_node_name, fuse_node));

            // Replace the input node to the new node
            auto nodes = node_outputs[embedding_end_at];
            for ( int i = 0; i < nodes.size(); i ++ ) {
                string stop_nodename = nodes.at(i);
                for (auto& node : *optimized_graph->mutable_node()) {
                    if ( node.name() != stop_nodename ) continue;

                    for ( int i = 0; i < node.input_size(); i ++ ) {
                        // Replace the old embedding node with the new one
                        if ( node.input(i) == embedding_end_at ) {
                            node.set_input(i, fuse_node_name);

                            break;
                        }
                    }

                    break;
                }
            }
            
	    if (enableEcFuseT && is_training_network)
	    {
                //set additional output
	        //additional output start from 1
	        uint32_t port = 0;
	        for (auto& out_tensor : additional_output_tensor) {
	            port++;

		    cout << "original tensor: " << out_tensor<<endl;

	        	//get the original tensor output nodes info 
                    auto node_infos = tensor_to_node[out_tensor];
                   
	            for (auto& node_i : node_infos) {
                    
                        std::size_t pos = node_i->node_name.find("grad");
		                      	
			if (pos == string::npos)
			{
			    continue;
			}	
                        
                        string s_port;
			if (port == 7){	
	    	            //select node share the same output as port 0
                            s_port = to_string(0);
			}
			else{
			    s_port = to_string(port);
			}

			node_mapping[node_i->node_name]->set_input(node_i->input_number, fuse_node_name + ":" + s_port);
	                
	                cout <<"----->to node: "<<node_i->node_name<<"\n changed to  new tensor name:"<<(fuse_node_name + ":" + s_port)<<endl; 	    
	            }

	        } 
	    }

            optimizer_idx ++;
        }

        printf(" [Done] - SparseEmbeddingOptimizer.\n");
       
    }
    
    if (enableEcFuseT && is_training_network)
    {
        printf("\n [Begin] - UnsortedSegmentSum fuse .\n");
        
	std::vector<string> unsortedSs_nodes;
        
	string node_device = "";
        for (auto& node : *(optimized_graph->mutable_node())) {
            string node_op = node.op();
            
	    // Record all nodes with 'UnsortedSegmentSum' type
            string node_name = node.name();

	    if ( node_op == "UnsortedSegmentSum" ) {
                unsortedSs_nodes.push_back(node_name);
                node_device = node.device();
             }      

     	}
	
        if ( unsortedSs_nodes.size() == 0 ) {
            printf(" [Over] - Target node is not found.\n");
            return Status::OK();
        }

        printf("    [Running] - Discovered %d unsorted_segment_sum.\n", unsortedSs_nodes.size());
        
	std::unordered_map<string, std::vector<string>> node_outputs;
        std::unordered_map<string, NodeDef*> node_mapping;
   

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
            
                if( node_outputs.count(prenode_name) > 0 ) {
                   node_outputs[prenode_name].push_back(node_name);
                } else {
                   std::vector<string> tmp_vector = {node_name};
                   node_outputs.insert(std::make_pair(prenode_name, tmp_vector));
                }
            }
        }



        // fuse
        GraphProperties properties(item);
        properties.InferStatically(false);

        int optimizer_idx = 0;
        for ( uint32_t i = 0; i < unsortedSs_nodes.size(); i ++ ) {
            
	    //we are trying to fuse four nodes with
	    //pattern 0 :
	    //root node: UnsortedSegmentSum
	    //input 0: GatherV2
	    //(input 1: Unique)
	    //input 2: stridedSlice
	    //...........input 0 of stridedSlice: Shape
            
	     
            bool pattern0_matched = false;
	    bool pattern1_matched = false;
            bool pattern2_matched = false;

	    NodeDef* root_node = node_mapping[unsortedSs_nodes[i]];
            //collect input tensors for new node
            vector<string> input_tensors;
	    vector<DataType> input_types;
	    
	    pattern0_matched = match_unsorted_segment_sum_pattern(0 /*pattern*/, properties, root_node, node_mapping, input_tensors, input_types); 
            
	    if (!pattern0_matched)
	    {
                //pattern 1 :
	        //root node: UnsortedSegmentSum
	        //input 0: Reshape
	        //(input 1: Unique:1
	        //input 2: stridedSlice
	        //...........input 0 of stridedSlice: Shape
                
	        pattern1_matched = match_unsorted_segment_sum_pattern(1 /*pattern*/, properties, root_node, node_mapping, input_tensors, input_types); 
			        
	    }	

            if (!pattern0_matched && !pattern1_matched)
	    {
		//pattern 2 :
	        //root node: UnsortedSegmentSum
	        //input 0: GatherV2
	        //input 1: reshape_fused
	        //input 2: stridedSlice
	        //...........input 0 of stridedSlice: Shape
                
	        pattern2_matched = match_unsorted_segment_sum_pattern(2 /*pattern*/, properties, root_node, node_mapping, input_tensors, input_types); 
		
                		    
	    }

            
            if ( !pattern0_matched && !pattern1_matched && !pattern2_matched)
	    {
	        continue;
	    }


            // Create the new node
            NodeDef* fuse_node = optimized_graph->add_node();
            string fuse_node_name;
            if (pattern0_matched || pattern2_matched)
	    {
                fuse_node_name = root_node->name() + "_fused1";
                fuse_node->set_op("UnsortedSegmentSumFused");
	    }
            else
	    {
	        fuse_node_name = root_node->name() + "_fused2";
                fuse_node->set_op("UnsortedSegmentSumFused2");
	    }

    
	    fuse_node->set_name(fuse_node_name);
            fuse_node->set_device(node_device);
            
	    assert(input_tesors.size() == input_types.size());
            
	    fuse_node->add_input(input_tensors[0]);
	    fuse_node->add_input(input_tensors[1]);
            (*fuse_node->mutable_attr())["Tdata"].set_type(input_types[0]);
            (*fuse_node->mutable_attr())["Tid"].set_type(input_types[1]);

            if( node_outputs.count(root_node->name()) == 0 ) {
                printf(" [Over] - %s has no output.", root_node->name().c_str());
                return Status::OK();
            }

            // Replace the input node to the new node
            auto nodes = node_outputs[root_node->name()];
            for ( uint32_t i = 0; i < nodes.size(); i ++ ) {
                string stop_nodename = nodes.at(i);
                for (auto& node : *(optimized_graph->mutable_node())) {
                    if ( node.name() != stop_nodename ) continue;

                    for ( uint32_t i = 0; i < node.input_size(); i ++ ) {
                        // Replace the old embedding node with the new one
                        if ( get_node_by_tensor(node.input(i)) == root_node->name() ) {
                            //todo: for multiple output
			    node.set_input(i, fuse_node_name);
			    printf("fused node output set!\n");
                            //break;
                        }
                    }
                    //break;
                }
            }

            optimizer_idx ++;
        }
   
	printf(" [Done] - UnsortedSegmentSum fuse %d nodes.\n", optimizer_idx);
    }

    return Status::OK();
}

void SparseEmbeddingOptimizer::Feedback(Cluster* /*cluster*/,
                                const GrapplerItem& /*item*/,
                                const GraphDef& /*optimized_graph*/,
                                double /*result*/) {
    // Nothing to do for LoopOptimizer.
}

}  // end namespace grappler
}  // namespace tensorflow
