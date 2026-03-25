# cmake/CustomScripts.cmake

# 函数：运行代码生成器
function(run_code_generator input_dir output_dir)
    file(GLOB_RECURSE INPUT_FILES ${input_dir}/*.proto)
    
    foreach(proto_file ${INPUT_FILES})
        get_filename_component(name ${proto_file} NAME_WE)
        set(output_file ${output_dir}/${name}.pb.cc)
        
        add_custom_command(
            OUTPUT ${output_file}
            COMMAND protoc --cpp_out=${output_dir} -I${input_dir} ${proto_file}
            DEPENDS ${proto_file}
            COMMENT "Generating C++ from ${proto_file}"
        )
        
        list(APPEND GENERATED_SOURCES ${output_file})
    endforeach()
    
    set(GENERATED_SOURCES ${GENERATED_SOURCES} PARENT_SCOPE)
endfunction()

# 函数：嵌入资源文件
function(embed_resources resource_dir output_cpp)
    add_custom_command(
        OUTPUT ${output_cpp}
        COMMAND python3 ${CMAKE_SOURCE_DIR}/scripts/embed_resources.py 
                ${resource_dir} ${output_cpp}
        DEPENDS ${CMAKE_SOURCE_DIR}/scripts/embed_resources.py}
        COMMENT "Embedding resources from ${resource_dir}"
    )
endfunction()