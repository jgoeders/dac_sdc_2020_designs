cmake_minimum_required(VERSION 3.10.2)
project(SkyNet)

set(CMAKE_CXX_STANDARD 14)

add_executable(SkyNet main.cpp SkyNet.cpp utils.cpp transform.cpp)