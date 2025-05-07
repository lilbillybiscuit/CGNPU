#include <iostream>
#include <vector>

int main() {
    int n;
    std::cin >> n;
    std::vector matrix1(n, std::vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> matrix1[i][j];
        }
    }
    std::vector matrix2(n, std::vector<int>(n));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cin >> matrix2[i][j];
        }
    }
    std::vector result(n, std::vector(n, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                result[i][j] += matrix1[i][k] * matrix2[k][j];
            }
        }
    }
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            std::cout << result[i][j];
            if (j < n - 1) std::cout << " ";
        }
        std::cout << std::endl;
    }
    return 0;
}