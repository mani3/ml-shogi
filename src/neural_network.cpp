#include "neural_network.h"


NeuralNetwork::NeuralNetwork(std::string model_path, unsigned int batch_size)
  : running(true),
    loop(nullptr),
    batch_size(batch_size) {

  tensorflow::SessionOptions session_options = tensorflow::SessionOptions();
  tensorflow::RunOptions run_options = tensorflow::RunOptions();
  tensorflow::Status status = tensorflow::LoadSavedModel(
    session_options, run_options, model_path, {tensorflow::kSavedModelTagServe}, &bundle);
  TF_CHECK_OK(status);

  this->loop = std::unique_ptr<std::thread>(new std::thread(NeuralNetwork::RunLoop, this));
}

NeuralNetwork::~NeuralNetwork() {
  this->running = false;
  this->loop->join();
}

std::future<NeuralNetwork::return_type> NeuralNetwork::Commit(Position* position) {
  std::vector<float> tensor = this->CreateInput(position);
  std::promise<NeuralNetwork::return_type> promise;
  auto ret = promise.get_future();
  {
    std::lock_guard<std::mutex> lock(this->lock);
    this->tasks.emplace(std::make_pair(tensor, std::move(promise)));
  }

  this->cv.notify_all();
  return ret;
}

void NeuralNetwork::Infer() {
  std::vector<std::vector<float>> states;
  std::vector<std::promise<return_type>> promises;

  bool timeout = false;

  while (states.size() < this->batch_size && !timeout) {
    std::unique_lock<std::mutex> lock(this->lock);

    if (this->cv.wait_for(lock, std::chrono::milliseconds(1), [this] { return this->tasks.size() > 0; })) {
      auto task = std::move(this->tasks.front());
      states.emplace_back(std::move(task.first));
      promises.emplace_back(std::move(task.second));

      this->tasks.pop();
    } else {
      std::cout << "timeout" << std::endl;
      timeout = true;
    }
  }

  if (states.size() == 0) {
    return;
  }

  tensorflow::Tensor inputs = MakeTensor(states);
  std::vector<tensorflow::Tensor> outputs;

  tensorflow::Status status = bundle.GetSession()->Run({{"serving_default_input_1:0", inputs}}, output_layer, {}, &outputs);
  TF_CHECK_OK(status);

  auto policy = outputs[0].flat<float>();
  std::vector<float> p_batch(policy.data(), policy.data() + policy.size());

  auto value = outputs[1].flat<float>();
  std::vector<float> v_batch(value.data(), value.data() + value.size());

  for (unsigned int i = 0; i < states.size(); i++) {
    auto p = std::vector<float>(p_batch.begin() + kPolicySize * i, p_batch.begin() + kPolicySize * (i + 1));
    auto v = std::vector<float>(v_batch.begin() + i, v_batch.begin() + (i + 1));
    return_type temp { std::move(p), std::move(v) };
    promises[i].set_value(std::move(temp));
  }
}

tensorflow::Tensor NeuralNetwork::MakeTensor(const std::vector<std::vector<float>> inputs) {
  const int batch_size = int(inputs.size());
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({batch_size, 9, 9, kChannelNum}));
  
  auto dst = tensor.flat<float>().data();
  for (auto v: inputs) {
    std::copy_n(v.begin(), v.size(), dst);
    dst += v.size();
  }
  return tensor;
}

std::vector<float> NeuralNetwork::CreateInput(const Position* position) {
  std::vector<float> tensor(this->kInputSize);

  const int move_num = position->gamePly();
  const int chanel_num = this->kChannelNum;
  
  for (Color c = Black; c < ColorNum; ++c) {
    Color color = c;
    if (position->turn() == White) {
      color = oppositeColor(c);
    }

    Bitboard bb[PieceTypeNum];
    for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
      bb[pt] = position->bbOf(pt, c);
      // bb[pt].printBoard();
    }

    // 先後で channel の位置が異なる
    const int start_index = color * (int(PieceTypeNum - 1) + HandPieceNum);

    for (PieceType pt = Pawn; pt < PieceTypeNum; ++pt) {
      for (Square sq = SQ11; sq < SquareNum; ++sq) {
			  Square square = sq;
			  if (position->turn() == White) {
				  square = SQ99 - sq;
		  	}

        if (bb[pt].isSet(sq)) {
          tensor[start_index + (this->kPyShogiPiece[pt] - 1) + chanel_num * this->kPyShogiSquare[square]] = 1;
			  }
		  }
		}

    // 持ち駒
    Hand hand = position->hand(c);
    for (HandPiece hp = HPawn; hp < HandPieceNum; ++hp) {
			u32 num = hand.numOf(hp);
      for (Square sq = SQ11; sq < SquareNum; ++sq) {
        tensor[start_index + (PieceTypeNum - 1) + hp + chanel_num * sq] = num;
      }
    }
  }

  // 手数
  for (Square sq = SQ11; sq < SquareNum; ++sq) {
    tensor[(chanel_num - 1) + chanel_num * sq] = move_num;
  }
  return tensor;
}
