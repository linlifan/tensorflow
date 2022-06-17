#include <vector>
#include <string>
#include <stdio.h>
#include <chrono>
#include <thread>
#include <mutex>

#include "tensorflow/cc/saved_model/loader.h"
#include "tensorflow/core/profiler/lib/profiler_session.h"
#include "tensorflow/core/profiler/rpc/client/capture_profile.h"

#include "tensorflow/core/platform/threadpool.h"

using Clock = std::chrono::high_resolution_clock;
using MS = std::chrono::duration<double, std::milli>;

int main(int argc, char* argv[]) {
  if (argc < 5) {
    printf("Usage: saved_model_inference PATH QPS BS RANGE [inter_op] [intra_op] [profile_flag]\n");
    return 0;
  }

  std::string model_path = argv[1];
  int QPS = atoi(argv[2]);
  printf("Target QPS: %d \n", QPS);
  int BS = atoi(argv[3]);
  printf("BS: %d \n", BS);
  int range = atoi(argv[4]);
  if (BS - range < 1) {
    range = BS - 1;
  }
  printf("Range: %d \n", range);

  tensorflow::SavedModelBundle bundle;
  tensorflow::SessionOptions session_options;
  tensorflow::RunOptions run_options;

  if (argc >= 7) {
    int inter_op = atoi(argv[5]);
    int intra_op = atoi(argv[6]);
    session_options.config.set_intra_op_parallelism_threads(intra_op);
    session_options.config.set_inter_op_parallelism_threads(inter_op);
  }

  int profile = 0;
  if (argc >= 8) {
    profile = atoi(argv[7]);
  }

  printf("intra_op: %d\n", session_options.config.intra_op_parallelism_threads());
  printf("inter_op: %d\n", session_options.config.inter_op_parallelism_threads());

  auto status = tensorflow::LoadSavedModel(session_options, run_options, model_path, {"serve"}, &bundle);

  if (!status.ok()) {
    printf("Failed to load model. Error message: %s \n", status.error_message().c_str());
    return -1;
  }

  auto session = bundle.GetSession();

  auto signature = bundle.GetSignatures().at("serving_default");

  std::vector<std::vector<std::pair<std::string, tensorflow::Tensor>>> input_data;
  int input_size = range * 2 + 1;
  input_data.resize(input_size);
  auto inputs = signature.inputs();
  printf("Inputs:\n");
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    auto tensor_info = it->second;
    printf("%s %s %s \n", tensor_info.name().c_str(), DataTypeString(tensor_info.dtype()).c_str(), tensor_info.tensor_shape().DebugString().c_str());
    for (int i = 0; i < input_size; i++) {
      tensorflow::TensorShape shape;
      int bs = BS - range + i;
      for (const auto& dim : tensor_info.tensor_shape().dim()) {
        shape.AddDim(dim.size() == -1 ? bs : dim.size());
      }
      input_data[i].push_back({tensor_info.name(), tensorflow::Tensor(tensor_info.dtype(), shape)});
    }
  }

  std::vector<std::string> output_nodes;
  auto outputs = signature.outputs();
  printf("Output nodes:\n");
  for (auto it = outputs.begin(); it != outputs.end(); it++) {
    output_nodes.push_back(it->second.name());
    printf("%s\n", it->second.name().c_str());
  }

  std::vector<tensorflow::Tensor> predictions;

  int warm = 1;
  for(int i = 0; i < warm; i++) {
    auto status = session->Run(input_data[rand() % input_size], output_nodes, {}, &predictions);
    if (!status.ok()) {
      printf("Failed to run. Error message: %s \n", status.error_message().c_str());
      return 0;
    }
  }

  printf("start benchmarking.\n");

  constexpr int AVG_SIZE = 100;
  double latencies[100];
  int idx = 0;
  std::mutex mu;

  auto SessionRun = [&]() {
    std::vector<tensorflow::Tensor> pred;

    int bs = rand() % input_size;
    
    Clock::time_point start = Clock::now();

    session->Run(input_data[bs], output_nodes, {}, &pred);

    Clock::time_point stop = Clock::now();
    MS diff = std::chrono::duration_cast<MS>(stop - start);
    printf("Latency: %f ms. BS: %d \n", diff.count(), BS - range + bs);

    std::lock_guard<std::mutex> guard(mu);
    if (idx >= AVG_SIZE) {
      double sum = 0.0f;
      for (int i = 0; i < AVG_SIZE; i++) {
        sum += latencies[i];
      }
      printf("Average latency of last %d session runs: %f ms\n", AVG_SIZE, sum / AVG_SIZE);
      idx = 0;
    }

    latencies[idx] = diff.count();
    idx++;
  };

  std::unique_ptr<tensorflow::ProfilerSession> profiler_session;

  int thread_num = 20;
  tensorflow::thread::ThreadPool pool(tensorflow::Env::Default(), "threadpool", thread_num);

  Clock::time_point t = Clock::now();
  for(int i = 0; i<10000; i++) {
    if (profile > 0) {
      if (i == QPS*10) {
        printf("start profiler\n");
        auto opt = tensorflow::ProfilerSession::DefaultOptions();
        profiler_session = tensorflow::ProfilerSession::Create(opt);
      } else if (i == QPS*11) {
        tensorflow::profiler::XSpace xspace;
        profiler_session->CollectData(&xspace);
        profiler_session.reset();
        tensorflow::profiler::ExportToTensorBoard(xspace, "logdir");
        printf("stop profiler\n");
      }
    }
    t += std::chrono::microseconds(1000*1000/QPS);
    std::this_thread::sleep_until(t);

    //std::thread t(SessionRun);
    //t.detach();
    pool.Schedule(SessionRun);
  }

  std::this_thread::sleep_for(std::chrono::seconds(1));

  return 0;
}

