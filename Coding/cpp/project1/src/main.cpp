#include <vector>
#include <deque>
#include <iostream>
#include "mylib.h"
using namespace std;
 // Create a static instance of MyLib to ensure the constructor is called

int main(int argc, char* argv[]) {
    std::vector<int> vec;
    std::deque<int> deq;

    // Example usage of vector
    vec.push_back(1);
    vec.push_back(2);
    vec.push_back(3);

    // Example usage of deque
    deq.push_back(1);
    deq.push_back(2);
    deq.push_back(3);
    
    auto res = MyLib::add(5, 3);
    auto res2 = MyLib::multiply(5, 3);
    auto vo = MyLib::VectorOps();
    auto res3 =  vo.sum(vec);
    auto res4 =  MyLib::VectorOps::mean(vec);
    std::cout<< "vector size: " << vec.size() << std::endl;
    std::cout<< "Result: " << res << std::endl;
    std::cout<< "Result2: " << res2 << std::endl;
    std::cout<< "Result3: " << res3 << std::endl;
    std::cout<< "Result4: " << res4 << std::endl;
    return 0;

}