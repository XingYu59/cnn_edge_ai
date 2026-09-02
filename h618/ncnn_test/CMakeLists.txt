cmake_minimum_required(VERSION 3.22)

project(h618_ncnn_test)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn REQUIRED)

# ncnn 预编译包的头文件在 <prefix>/include/ncnn/,
# 但 INTERFACE_INCLUDE 只指到 include/ncnn (期望 "net.h" 风格)。
# 这里显式加 include 根, 以支持 #include <ncnn/net.h>。
set(NCNN_INCLUDE_ROOT $ENV{HOME}/RK3566_dev/ncnn-20260526-android-vulkan/armeabi-v7a/include)

add_executable(h618_ncnn_test
    main.cpp
)
target_include_directories(h618_ncnn_test PRIVATE ${NCNN_INCLUDE_ROOT})
target_link_libraries(h618_ncnn_test ncnn)

add_executable(infer_mini
    infer_mini.cpp
)
target_include_directories(infer_mini PRIVATE ${NCNN_INCLUDE_ROOT})
target_link_libraries(infer_mini ncnn)
