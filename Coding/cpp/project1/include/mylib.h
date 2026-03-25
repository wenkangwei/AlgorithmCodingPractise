#if !defined(MYLIB_H)
#define MYLIB_H
#include <iostream>
#include <vector>

// Your library code here
namespace MyLib {
    // Anonymous namespace to limit the scope of myLibInstance
    int add(int a, int b){
        return a + b;
    }

    int multiply(int a, int b){
        return a * b;
    }


    class VectorOps{
        public:
            static void printVector(const std::vector<int>& vec) {
                std::cout << "Vector contents: ";
                for (const auto& elem : vec) {
                    std::cout << elem << " ";
                }
                std::cout << std::endl;
            }
            int sum(std::vector<int> vec){
                int total = 0;
                for (const auto& num : vec) {
                    total += num;
                }
                return total;
            }

            static int mean(std::vector<int> vec){
                int total = 0;
                for (const auto& num : vec) {
                    total += num;
                }
                return total/vec.size();
            }
    };
}







#endif // MYLIB_H