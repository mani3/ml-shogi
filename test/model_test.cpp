#include <iostream>
#include <string>

// #include "neural_network.h"

#include <common.hpp>
#include <bitboard.hpp>
#include <init.hpp>
#include <position.hpp>
#include <hand.hpp>
#include <usi.hpp>
#include <thread.hpp>
#include <tt.hpp>
#include <search.hpp>

#include <move.hpp>
#include <generateMoves.hpp>

using namespace std;

int main(int argc, char **argv) {

  if (argc < 2) {
    std::cout << "Error missing parameter: " << argv[0] << " <tflite_model_path>" << std::endl;
    exit(0);
  }

  std::string model_path = argv[1];
  // NeuralNetwork model(model_path, 1);

  initTable();
  Position::initZobrist();

  auto s = std::unique_ptr<Searcher>(new Searcher());
  s->init();

  auto th = std::unique_ptr<Thread>(new Thread(s->thisptr));
  // auto position = std::unique_ptr<Position>(new Position(DefaultStartPositionSFEN, th.get(), s->thisptr));
  // auto res = model.commit(position.get()).get();

  Position position(DefaultStartPositionSFEN, th.get(), s->thisptr);

  // auto res = model.Commit(&position).get();

  // auto p = res[0];
  // auto v = res[1];

  // std::for_each(p.begin(), p.end(), [](float x) { std::cout << x << ","; });
  // std::cout << std::endl;
  // std::for_each(v.begin(), v.end(), [](float x) { std::cout << x << ","; });
  // std::cout << std::endl;
  return 0;
}
