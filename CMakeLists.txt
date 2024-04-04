cmake_minimum_required(VERSION 3.10)

project(NN)
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)



set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wformat-nonliteral -Wcast-align -Wpointer-arith -Wmissing-declarations -Winline -Wundef -Wcast-qual -Wshadow -Wwrite-strings -Wno-unused-parameter -Wfloat-equal -pedantic -ansi")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/src)


SET(TARGETUL "tests")

add_executable(${TARGETUL} ${TARGETUL}.cpp NeuralNetwork.cpp)
target_link_libraries(${TARGETUL} raylib GL m pthread dl rt X11)
target_compile_options(${TARGETUL} PRIVATE -O3)