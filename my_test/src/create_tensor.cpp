#include <torch/torch.h>
#include <iostream>

int main() {
    torch::Tensor zeros_tensor = torch::zeros({2, 3}); // 2x3 tensor of zeros
    std::cout << "Zeros Tensor:\n" << zeros_tensor << std::endl;

    return 0;
}