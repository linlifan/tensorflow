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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FEATURE_COLUMN_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FEATURE_COLUMN_OPTIMIZER_H_

//#include "tensorflow/core/grappler/optimizers/sparse_embedding_optimizer.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {

class FeatureColumnOptimizer {
 public:
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
    if (position != string::npos)
        tensor_name.erase(tensor_name.begin()+ position, tensor_name.end());
    if (tensor_name[0] == '^')
        tensor_name.erase(tensor_name.begin());

    return tensor_name;
  }

  NodeDef* skip_identity(std::unordered_map<string, NodeDef*>& node_mapping, string node_name) {
    NodeDef* node = node_mapping[node_name];
    while (node->op() == "Identity") {
        node_name = get_node_by_tensor(node->input(0));
        node = node_mapping[node_name];
    }
    return node;
  }

  NodeDef* find_output_node(std::unordered_map<string, std::vector<string>>& node_outputs,
                    std::unordered_map<string, NodeDef*>& node_mapping,
                    string node_name, std::vector<string>& target_ops) {
    if (node_outputs.count(node_name) == 0) return NULL;

    auto nodes = node_outputs[node_name];
    for (int i = 0; i < nodes.size(); i++) {
        string outnode_name = nodes.at(i);
        NodeDef* tmp_outnode = node_mapping[outnode_name];
        if (find(target_ops.begin(), target_ops.end(), tmp_outnode->op()) != target_ops.end())
            return tmp_outnode;
    }
    return NULL;
  }

  int check_graph(GraphDef* optimized_graph, std::vector<string>& target_node_names,
                string& node_device, string feature_type) {
    for (auto& node : *optimized_graph->mutable_node()) {
        if (node.op() == feature_type) {
          printf(" [Over] - Already fused.\n");
          return 0;
        }

        // Record all nodes with 'GatherNd' type
        if (node.op() == "StringToHashBucketFast") {
          string node_name = node.name();
          target_node_names.push_back(node_name);
          node_device = node.device();
        }
    }

    if (target_node_names.size() == 0) {
        printf(" [Over] - Target node is not found.\n");
        return 0;
    }

    return 1;
  }

  void collect_the_common_information(GraphDef* optimized_graph,
                  std::unordered_map<string, std::vector<string>>& node_outputs,
                  std::unordered_map<string, NodeDef*>& node_mapping) {
    for (auto& node : *optimized_graph->mutable_node()) {
        string node_name = node.name();
        node_mapping.insert(std::make_pair(node_name, &node));

        int input_num = node.input_size();
        if (input_num == 0) continue;

        for (int i = 0; i < input_num; i++) {
            string prenode_name = get_node_by_tensor(node.input(i));

            if (node_outputs.count(prenode_name) > 0) {
                node_outputs[prenode_name].push_back(node_name);
            } else {
                std::vector<string> tmp_vector = {node_name};
                node_outputs.insert(std::make_pair(prenode_name, tmp_vector));
            }
        }
    }

  }

  int to_sparse_input(GraphProperties& properties,
                  std::unordered_map<string, std::vector<string>>& node_outputs,
                  std::unordered_map<string, NodeDef*>& node_mapping,
                  string gatherNd_node_name,
                  string& input_name, string& weight_name, string &indices_name, string& shape_name,
                  DataType& Tid_value, DataType& Tweight_value) {

    NodeDef* values_node = node_mapping[gatherNd_node_name];
    if (values_node->op() != "GatherNd") {
        printf(" [Over] - Cannot find `GatherNd` for %s.\n", gatherNd_node_name.c_str());
        return 0;
    }
    indices_name = get_node_by_tensor(values_node->input(1));
    NodeDef* indices_node = node_mapping[indices_name];
    if (indices_node->op() != "Where") {
        printf(" [Over] - Cannot find where_node for %s.\n", indices_name.c_str());
        return 0;
    }
    string notEqual_name = get_node_by_tensor(indices_node->input(0));
    NodeDef* notEqual_node = node_mapping[notEqual_name];
    if (notEqual_node->op() != "NotEqual") {
        printf(" [Over] - Cannot find notEqual_node for %s.\n", notEqual_name.c_str());
        return 0;
    }

    // Get the real input
    input_name = get_node_by_tensor(notEqual_node->input(0));
    weight_name = notEqual_node->input(1);

    std::vector<string> target_ops = {"Shape"};
    NodeDef* shape_node = find_output_node(node_outputs, node_mapping, input_name, target_ops);
    if (!shape_node) {
        printf(" [Over] - Cannot find Shape node for %s.\n", input_name.c_str());
        return 0;
    }
    shape_name = shape_node->name();

    Tid_value = GetInputType(properties, notEqual_node, 0);
    Tweight_value = GetInputType(properties, notEqual_node, 1);

    return 1;
  }
};

}  // end namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_FEATURE_COLUMN_OPTIMIZER_H_
