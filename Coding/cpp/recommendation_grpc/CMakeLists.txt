cmake_minimum_required(VERSION 3.5)

project(RecommendationSystem VERSION 1.0 LANGUAGES CXX)

# 设置 C++ 标准为 C++14 (gRPC需要至少C++11，推荐C++14或更高)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 添加头文件搜索路径
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_BINARY_DIR}/proto)

# 查找 Protobuf
find_package(Protobuf REQUIRED)
include_directories(${Protobuf_INCLUDE_DIRS})

# 查找 gRPC - 尝试多种方法
# 方法1: CONFIG模式（推荐的现代方式）
find_package(gRPC CONFIG QUIET)

# 如果CONFIG模式失败，尝试MODULE模式
if(NOT gRPC_FOUND)
    find_package(gRPC QUIET)
endif()

# 如果还是找不到，使用pkg-config
if(NOT gRPC_FOUND)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(GRPC REQUIRED grpc++ grpc)
    if(GRPC_FOUND)
        include_directories(${GRPC_INCLUDE_DIRS})
    endif()
endif()

# 如果所有方法都失败，报错
if(NOT gRPC_FOUND AND NOT GRPC_FOUND)
    message(FATAL_ERROR "gRPC not found. Please install gRPC:\n"
            "  Ubuntu/Debian: sudo apt-get install libgrpc++-dev libgrpc-dev\n"
            "  Or build from source: https://github.com/grpc/grpc")
endif()

# 定义protobuf文件
set(PROTO_FILES
    ${PROJECT_SOURCE_DIR}/proto/recommendation.proto
)

# 生成protobuf代码
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILES})

# 生成gRPC代码
# 检查grpc_generate_cpp函数是否可用
if(COMMAND grpc_generate_cpp)
    grpc_generate_cpp(GRPC_SRCS GRPC_HDRS ${PROTO_FILES})
else()
    # 如果grpc_generate_cpp不可用，使用protoc生成
    message(STATUS "grpc_generate_cpp not available, using protoc directly")
    find_program(PROTOC protoc)
    find_program(GRPC_CPP_PLUGIN grpc_cpp_plugin)
    if(NOT PROTOC)
        message(FATAL_ERROR "protoc not found. Please install protobuf-compiler")
    endif()
    if(NOT GRPC_CPP_PLUGIN)
        message(FATAL_ERROR "grpc_cpp_plugin not found. Please install protobuf-compiler-grpc")
    endif()
    
    # 确保输出目录存在
    file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/proto)
    
    # 为每个proto文件生成gRPC代码
    foreach(proto_file ${PROTO_FILES})
        get_filename_component(proto_name ${proto_file} NAME_WE)
        set(grpc_src "${CMAKE_CURRENT_BINARY_DIR}/proto/${proto_name}.grpc.pb.cc")
        set(grpc_hdr "${CMAKE_CURRENT_BINARY_DIR}/proto/${proto_name}.grpc.pb.h")
        
        add_custom_command(
            OUTPUT ${grpc_src} ${grpc_hdr}
            COMMAND ${PROTOC}
            ARGS --grpc_out=${CMAKE_CURRENT_BINARY_DIR}/proto
                 --plugin=protoc-gen-grpc=${GRPC_CPP_PLUGIN}
                 --cpp_out=${CMAKE_CURRENT_BINARY_DIR}/proto
                 -I${CMAKE_CURRENT_SOURCE_DIR}/proto
                 ${proto_file}
            DEPENDS ${proto_file}
            COMMENT "Generating gRPC code from ${proto_file}"
            VERBATIM
        )
        
        list(APPEND GRPC_SRCS ${grpc_src})
        list(APPEND GRPC_HDRS ${grpc_hdr})
    endforeach()
endif()

# 添加生成的源文件到项目中
add_custom_target(proto_gen ALL DEPENDS ${PROTO_SRCS} ${GRPC_SRCS})

# 创建服务器可执行文件
add_executable(recommendation_server
    src/recommendation_server.cpp
    src/recommendation_service_impl.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# 根据找到的gRPC方式选择链接库
if(TARGET gRPC::grpc++)
    target_link_libraries(recommendation_server
        gRPC::grpc++
        ${Protobuf_LIBRARIES}
    )
else()
    target_link_libraries(recommendation_server
        ${GRPC_LIBRARIES}
        ${Protobuf_LIBRARIES}
    )
endif()

# 创建客户端可执行文件
add_executable(recommendation_client
    src/recommendation_client.cpp
    ${PROTO_SRCS}
    ${GRPC_SRCS}
)

# 根据找到的gRPC方式选择链接库
if(TARGET gRPC::grpc++)
    target_link_libraries(recommendation_client
        gRPC::grpc++
        ${Protobuf_LIBRARIES}
    )
else()
    target_link_libraries(recommendation_client
        ${GRPC_LIBRARIES}
        ${Protobuf_LIBRARIES}
    )
endif()

# 确保proto文件先生成
add_dependencies(recommendation_server proto_gen)
add_dependencies(recommendation_client proto_gen)

# 设置输出目录
set_target_properties(recommendation_server PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)

set_target_properties(recommendation_client PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}
)

# 可选：添加编译选项
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(recommendation_server PRIVATE -g -O0)
    target_compile_options(recommendation_client PRIVATE -g -O0)
else()
    target_compile_options(recommendation_server PRIVATE -O3)
    target_compile_options(recommendation_client PRIVATE -O3)
endif()

# 打印配置信息
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Protobuf version: ${Protobuf_VERSION}")
message(STATUS "gRPC found: ${gRPC_FOUND}")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")