cmake_minimum_required(VERSION 3.22)

project(h618_ncnn_bench)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ncnn REQUIRED)

set(NCNN_INCLUDE_ROOT $ENV{HOME}/RK3566_dev/ncnn-20260526-android-vulkan/armeabi-v7a/include)

add_executable(h618_ncnn_bench
    main.cpp
)
target_include_directories(h618_ncnn_bench PRIVATE ${NCNN_INCLUDE_ROOT})
target_link_libraries(h618_ncnn_bench ncnn)
