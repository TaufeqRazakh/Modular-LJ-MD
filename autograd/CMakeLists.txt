cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(autograd)

find_package(Torch REQUIRED)

add_executable(autograd autograd.cpp)
target_link_libraries(autograd "${TORCH_LIBRARIES}")
set_property(TARGET autograd PROPERTY CXX_STANDARD 11)
