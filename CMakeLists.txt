cmake_minimum_required(VERSION 3.0)

set( CMAKE_CXX_COMPILER /usr/bin/g++-8 )
set( CMAKE_CXX_FLAGS "-std=c++17 -O2 -Wall -Wextra -Wconversion")

message(STATUS "${CMAKE_CXX_FLAGS}")

set(STAN_MATH_DIR ${CMAKE_CURRENT_SOURCE_DIR}/math)
set(EIGEN_DIR ${CMAKE_CURRENT_SOURCE_DIR}/math/lib/eigen_3.3.3)
#set(BOOST_DIR ${CMAKE_CURRENT_SOURCE_DIR}/math/lib/boost_1.66.0)
set(CVODES_DIR ${CMAKE_CURRENT_SOURCE_DIR}/math/lib/cvodes-3.1.0/include)

include_directories(SYSTEM  #以下のライブラリでのwarningの抑制
    ${STAN_MATH_DIR}
    ${EIGEN_DIR}
    #${BOOST_DIR}
    ${CVODES_DIR}
    )

add_executable(main main.cpp)

