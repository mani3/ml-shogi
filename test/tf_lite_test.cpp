#include <iostream>
#include <string>

#include "tensorflow/lite/model.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"


using namespace std;

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "Error missing parameter: " << argv[0] << " <tflite_model_path>" << std::endl;
    exit(0);
  }

  const char *model_path = argv[1];
  tflite::StderrReporter error_reporter;
  std::unique_ptr<tflite::FlatBufferModel> model = tflite::FlatBufferModel::BuildFromFile(model_path, &error_reporter);

  if (!model) {
    std::cout << "Failed to load model: " << model_path << std::endl;
    exit(0);
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  std::unique_ptr<tflite::Interpreter> interpreter;
  tflite::InterpreterBuilder(*model.get(), resolver)(&interpreter);

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    std::cout << "Failed to allocate tensors on interpreter: " << model_path << std::endl;  
  }

  int index = 0;
  float* input = interpreter->typed_input_tensor<float>(index);
  *input = 1.0f;

  if (interpreter->Invoke() != kTfLiteOk) {
    std::cout << "Failed to invoke!: " << model_path << std::endl; 
  }

  float* output = interpreter->typed_output_tensor<float>(index);
  
  int POLICY_SIZE = 9 * 9 * 27;
  std::vector<float> v {output, output + POLICY_SIZE};
  float* value = interpreter->typed_output_tensor<float>(1);
  
  for (int i = 0; i < POLICY_SIZE; i++) {
    std::cout << "Policy: " << i << "=" << output[i] << ", " << v[i] << std::endl;
  }
  
  std::cout << "Value: " << value[0]  << std::endl;
  return 0;
}
