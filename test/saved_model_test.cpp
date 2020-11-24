#include <iostream>

#include <tensorflow/core/framework/tensor.h>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

using namespace std;
using namespace tensorflow;

const int CHANNEL_SIZE = 43;

tensorflow::Tensor ConcatInputs(const std::vector<std::vector<float>> inputs) {
  Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({int(inputs.size()), 9, 9, CHANNEL_SIZE}));
  
  auto dst = tensor.flat<float>().data();
  for (auto v: inputs) {
    std::copy_n(v.begin(), v.size(), dst);
    dst += v.size();
  }
  return tensor;
}

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "Error missing parameter: " << argv[0] << " <tflite_model_path>" << std::endl;
    exit(0);
  }
  std::string export_dir = argv[1];

  SavedModelBundleLite bundle;
  SessionOptions session_options = SessionOptions();
  RunOptions run_options = RunOptions();
  Status status = LoadSavedModel(session_options, run_options, export_dir, {kSavedModelTagServe}, &bundle);

  std::vector<std::vector<float>> values { 
    std::vector<float>(9 * 9 * CHANNEL_SIZE, 0.f), 
    std::vector<float>(9 * 9 * CHANNEL_SIZE, 1.f), 
    std::vector<float>(9 * 9 * CHANNEL_SIZE, 2.f),
  };

  tensorflow::Tensor inputs = ConcatInputs(values);
  
  std::vector<Tensor> outputs;
  const std::vector<string> output_layer = {
    "StatefulPartitionedCall:0", 
    "StatefulPartitionedCall:1",
  };
  Status runStatus = bundle.GetSession()->Run({{"serving_default_input_1:0", inputs}}, output_layer, {}, &outputs);
  std::cout << "status: " << runStatus << std::endl;

  std::cout << "==========================" << std::endl;

  Tensor policy = outputs[0];

  const int POLICY_SIZE = 9 * 9 * 27;

  auto size = policy.flat<float>().size();
  auto p = policy.flat<float>().data();

  std::cout << policy.flat<float>().data() << std::endl;
  std::vector<float> res(p, p + size);

  std::vector<std::vector<float>> policies;
  for (int i = 0; i < values.size(); i++) {
    auto out = std::vector<float>(res.begin() + POLICY_SIZE * i, res.begin() + POLICY_SIZE * (i + 1));
    policies.push_back(out);
  }

  for (int i = 0; i < values.size(); i++) {
    for (auto const& c : policies[i]) {
      std::cout << c << ' ';
    }
    std::cout << std::endl;  
  }
  
  // std::cout << policy.vec<float>()[0] << std::endl;
  // std::cout << policy[0].flat<float>().size() << std::endl;

  std::cout << "==========================" << std::endl;
  
  Tensor value = outputs[1];
  std::cout << value.flat<float>() << std::endl;

  return 0;
}
