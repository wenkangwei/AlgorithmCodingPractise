# 文档参考
文档参考以下教程进行简化：
-  https://www.runoob.com/cmake/cmake-build-flow.html
- 点击链接查看和 Kimi 的对话 https://www.kimi.com/share/19d246aa-4562-888f-8000-0000673d90ac

# 前置包安装
~~~bash
#Linux
sudo apt-get install cmake make 
~~~

# CMake构建流程
参考： https://www.runoob.com/cmake/cmake-build-flow.html

- **创建构建目录**：保持源代码目录整洁。
    - 创建src, include 等子目录
- **使用 CMake 生成构建文件**：配置项目并生成适合平台的构建文件。
    - 新建一个CMakeLists.txt文件，用于cmake生成不同平台的构建文件，比如 Makefile, Ninjia等
- **编译和构建**：使用生成的构建文件执行编译和构建。
    - 基于生成的构建文件，用make， ninja等工具对c++项目编译
- **清理构建文件**：删除中间文件和目标文件。
    - 构建过程中生成的非执行的中间文件比如object文件 不用的话进行清理
- **重新配置和构建**：处理项目设置的更改。
    - 修改CMakeLists.txt文件后，重新执行cmake生成新的构建文件，然后进行编译 



# 创建目录
~~~bash
mkdir project1
cd project1
mkdir -p src include tests scripts cmake build
touch CMakeLists.txt README.md
~~~

## 目录说明
~~~bash
my_vscode_cmake/
├── cmake/
│   ├── CustomScripts.cmake
│   └── FindDependencies.cmake
├── scripts/
│   ├── build.sh              # 一键构建脚本(从cmake到make构造可执行文件)
│   ├── pre_build.sh          # 预构建检查
│   ├── generate_version.sh   # 版本生成
│   └── package.sh            # 打包脚本
├── include/
│   └── myproject/
│       └── config.h.in       # 配置模板
├── src/                      # 程序入口文件和各种库文件
│   └── main.cpp
├── tests/                    #测试文件
│   └── CMakeLists.txt
└── CMakeLists.txt            # 项目cmake文件
~~~



# 编写自己的CMakeFile.txt 文件

~~~
cmake_minimum_required(VERSION 3.10)   # 指定最低 CMake 版本

project(MyProject VERSION 1.0)          # 定义项目名称和版本

# 设置 C++ 标准为 C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)

# 添加源文件
add_library(MyLib src/mylib.cpp)        # 创建一个库目标，名称叫MyLib
add_executable(MyExecutable src/main.cpp)  # 创建一个可执行文件目标 MyExecutable

# 链接库到可执行文件
target_link_libraries(MyExecutable MyLib)
~~~

更详细例子说明：

~~~txt
cmake_minimum_required(VERSION 3.20)
project(MyProject VERSION 1.0.0 LANGUAGES CXX)

# ========== 基础配置 ==========
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)  # 生成 compile_commands.json

# 构建类型默认 Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# 输出目录
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# ========== 编译选项 ==========
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        add_compile_options(-g -O0)
    else()
        add_compile_options(-O3)
    endif()
endif()

# ========== 源文件 ==========
file(GLOB_RECURSE SOURCES "src/*.cpp")
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE 
    ${CMAKE_SOURCE_DIR}/include
)

# ========== 测试 ==========
enable_testing()
add_subdirectory(tests)
~~~

# 编译项目

~~~bash
cd ./build
bash build.sh
# 详细指定参数： 输出版本， 安装目录位置 是否测试case
bash build.sh debug /usr/local  --test
~~~