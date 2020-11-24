#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include <iostream>
#include <future>
#include <memory>
#include <queue>
#include <string>
#include <utility>
#include <vector>

#include <tensorflow/core/framework/tensor.h>

#include <tensorflow/cc/ops/standard_ops.h>
#include <tensorflow/cc/client/client_session.h>
#include <tensorflow/cc/saved_model/loader.h>
#include <tensorflow/cc/saved_model/tag_constants.h>

#include <common.hpp>
#include <bitboard.hpp>
#include <move.hpp>
#include <hand.hpp>
#include <generateMoves.hpp>
#include <position.hpp>

class NeuralNetwork {

 public:
  using return_type = std::vector<std::vector<float>>;

  NeuralNetwork(std::string model_path, unsigned int batch_size);
  ~NeuralNetwork();

  std::future<return_type> Commit(Position* position);

  void SetBatchSize(unsigned int batch_size) {
    this->batch_size = batch_size;
  };

 private:
  using task_type = std::pair<std::vector<float>, std::promise<return_type>>;

  void Infer();
  std::vector<float> CreateInput(const Position* position);
  tensorflow::Tensor MakeTensor(const std::vector<std::vector<float>> inputs);

  bool running;
  std::unique_ptr<std::thread> loop;

  static void RunLoop(NeuralNetwork* model) {
    while (model->running) {
      model->Infer();
    }
  };

  std::queue<task_type> tasks;
  std::mutex lock;
  std::condition_variable cv;

  std::unique_ptr<tensorflow::SavedModelBundleLite> bundle;

  unsigned int batch_size;

  const std::vector<std::string> output_layer = {
    "StatefulPartitionedCall:0", 
    "StatefulPartitionedCall:1",
  };
  
  const int kChannelNum = 43; // 28 + 14 + 1;
  const int kInputSize  = 9 * 9 * kChannelNum; // 9 x 9 x 43
  const int kPolicySize = 9 * 9 * 27;

  const int kPyShogiSquare[81] = {
    8, 17, 26, 35, 44, 53, 62, 71, 80,
    7, 16, 25, 34, 43, 52, 61, 70, 79,
    6, 15, 24, 33, 42, 51, 60, 69, 78,
    5, 14, 23, 32, 41, 50, 59, 68, 77,
    4, 13, 22, 31, 40, 49, 58, 67, 76,
    3, 12, 21, 30, 39, 48, 57, 66, 75,
    2, 11, 20, 29, 38, 47, 56, 65, 74,
    1, 10, 19, 28, 37, 46, 55, 64, 73,
    0,  9, 18, 27, 36, 45, 54, 63, 72,
  };

  const int kPyShogiPiece[15] = {
    0, 1, 2, 3, 4, 6, 7, 5, 8, 9, 10, 11, 12, 14, 13
  };
};

#endif /* NEURAL_NETWORK_H_ */
