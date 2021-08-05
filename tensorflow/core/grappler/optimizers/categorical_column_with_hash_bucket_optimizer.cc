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

#include "tensorflow/core/grappler/optimizers/categorical_column_with_hash_bucket_optimizer.h"

#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/mutable_graph_view.h"
#include "tensorflow/core/grappler/op_types.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/lib/core/errors.h"

namespace tensorflow {
namespace grappler {
Status CategoricalColumnWithHashBucketOptimizer::Optimize(Cluster* cluster, const GrapplerItem& item,
                                GraphDef* optimized_graph) {
    *optimized_graph = item.graph;
    const char* env_p = std::getenv("FC_FUSE");
    if (env_p != NULL && env_p[0] == '0') {
       LOG(INFO) << "Ignore the optimization as FC_FUSE=0";
        return Status::OK();
    }

    printf("\n [Begin] - CategoricalColumnWithHashBucketOptimizer.\n");

    string node_device = "";
    std::vector<string> target_node_names;
    if (!check_graph(optimized_graph, target_node_names, node_device, "CategoricalColumnWithHashBucket"))
        return Status::OK();

    printf("    [Running] - Discovered %d subgraphs to be merged.\n", target_node_names.size());

    // collect the common information
    std::unordered_map<string, std::vector<string>> node_outputs;
    std::unordered_map<string, NodeDef*> node_mapping;
    collect_the_common_information(optimized_graph, node_outputs, node_mapping);

    // fuse
    GraphProperties properties(item);
    properties.InferStatically(false);

   int optimizer_idx = 0;
    for (string hash_name : target_node_names) {
        string input_name, weight_name, indices_name, shape_name;
        DataType Tid_value, Tweight_value;

        NodeDef* hash_node = node_mapping[hash_name];
        string values_name = get_node_by_tensor(hash_node->input(0));
        if (!to_sparse_input(properties, node_outputs, node_mapping, values_name,
                input_name, weight_name, indices_name, shape_name,
                Tid_value, Tweight_value)) continue;

        // Create the new node
       string fuse_node_name = "categorical_column_with_hash_bucket_fused_" + std::to_string(optimizer_idx);
        auto num_buckets = hash_node->attr().at("num_buckets").i();

        NodeDef* fuse_node = optimized_graph->add_node();
        fuse_node->set_op("CategoricalColumnWithHashBucket");
        fuse_node->set_name(fuse_node_name);
        fuse_node->set_device(node_device);
        fuse_node->add_input(input_name);
        fuse_node->add_input(weight_name);
        (*fuse_node->mutable_attr())["Tid"].set_type(Tid_value);
        (*fuse_node->mutable_attr())["Tweight"].set_type(Tweight_value);
        (*fuse_node->mutable_attr())["num_buckets"].set_i(num_buckets);
        (*fuse_node->mutable_attr())["feature_column_type"].set_s("CategoricalColumnWithHashBucket");

        if (node_outputs.count(hash_name) == 0) {
           printf(" [Over] - %s has no output.", hash_name.c_str());
            return Status::OK();
        }
        if (node_outputs.count(indices_name) == 0) {
            printf(" [Over] - %s has no output.", indices_name.c_str());
            return Status::OK();
        }
        if (node_outputs.count(shape_name) == 0) {
            printf(" [Over] - %s has no output.", shape_name.c_str());
            return Status::OK();
        }

        // Replace the input node to the new node
        std::vector<string> output_node_names;
        for (string stop_nodename : node_outputs[hash_name])
            output_node_names.push_back(stop_nodename);
        for (string stop_nodename : node_outputs[indices_name])
            output_node_names.push_back(stop_nodename);
        for (string stop_nodename : node_outputs[shape_name])
           output_node_names.push_back(stop_nodename);

        for (string stop_nodename : output_node_names) {
            for (auto& node : *optimized_graph->mutable_node()) {
                if (node.name() != stop_nodename) continue;

                for (int input_id = 0; input_id < node.input_size(); input_id++) {
                    // Replace the old embedding node with the new one
                    if (node.input(input_id) == hash_name) {
                        node.set_input(input_id, fuse_node_name);
                        break;
                    } else if (node.input(input_id) == indices_name) {
                        node.set_input(input_id, fuse_node_name + ":1");
                        break;
                    } else if (node.input(input_id) == shape_name) {
                        node.set_input(input_id, fuse_node_name + ":2");
                        break;
                    }
                }
            }
       }

        optimizer_idx++;
    }

    printf(" [Done] - CategoricalColumnWithHashBucketOptimizer.\n");
    return Status::OK();
}

void CategoricalColumnWithHashBucketOptimizer::Feedback(Cluster* /*cluster*/,
                                const GrapplerItem& /*item*/,
                                const GraphDef& /*optimized_graph*/,
                                double /*result*/) {
    // Nothing to do for LoopOptimizer.
}
}  // end namespace grappler
}  // namespace tensorflow

