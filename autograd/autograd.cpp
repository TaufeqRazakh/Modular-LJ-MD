#include <torch/torch.h>
#include <iostream>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/autograd/function.h>

int main() {
  torch::Tensor tensor = torch::eye(3);
  std::cout << tensor << std::endl;

  torch::Tensor a = torch::ones({2, 2}, torch::requires_grad());
  torch::Tensor b = torch::randn({2,2});
  auto c = a + b;
  c.backward();
  std::cout << c << std::endl;
}
