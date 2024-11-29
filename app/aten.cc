#include <ATen/ATen.h>
#include <iostream>

int main() {
    // Create a tensor
    at::Tensor tensor = at::rand({3, 3});  // 3x3 matrix with random values

    // Print the tensor
    std::cout << "Random Tensor:" << std::endl;
    std::cout << tensor << std::endl;

    // Perform operations
    at::Tensor result = tensor + 2;  // Add 2 to every element
    std::cout << "\nTensor after adding 2:" << std::endl;
    std::cout << result << std::endl;

    // Matrix multiplication
    at::Tensor mat1 = at::rand({3, 3});
    at::Tensor mat2 = at::rand({3, 3});
    at::Tensor product = at::matmul(mat1, mat2);
    std::cout << "\nMatrix Multiplication Result:" << std::endl;
    std::cout << product << std::endl;

    // Access tensor elements
    float value = tensor[0][0].item<float>();
    std::cout << "\nFirst element of original tensor: " << value << std::endl;

    return 0;
}