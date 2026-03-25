
#include "../include/mylib.h"

int MyLib::add(int a, int b) {
        return a + b;
}
int MyLib::multiply(int a, int b) {
    return a * b;
}

// int MyLib::VectorOps::sum(std::vector<int> vec) {
//     int total = 0;
//     for (const auto& num : vec) {
//         total += num;
//     }
//     return total;
// }