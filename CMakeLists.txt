cmake_minimum_required(VERSION 3.5)

project(deep_cyber VERSION 0.0.0)

# the library itself
add_library(${CMAKE_PROJECT_NAME} SHARED include/int.h src/tensor.c include/tensor.h include/deep_cyber.h src/conv2d.c src/dense.c src/activation.c src/pooling.c)
target_include_directories(${CMAKE_PROJECT_NAME} INTERFACE include)

# CIFAR10
add_library(${CMAKE_PROJECT_NAME}_cifar10 SHARED cifar10/cifar10.h cifar10/cifar10.c cifar10/cifar10_weights.h cifar10/c1b.c cifar10/c1w.c cifar10/c2b.c cifar10/c2w.c cifar10/c3b.c cifar10/c3w.c cifar10/c4b.c cifar10/c4w.c cifar10/d1b.c cifar10/d1w.c cifar10/d2b.c cifar10/d2w.c)
target_include_directories(${CMAKE_PROJECT_NAME}_cifar10 INTERFACE cifar10)
target_link_libraries(${CMAKE_PROJECT_NAME}_cifar10 PUBLIC ${CMAKE_PROJECT_NAME})

# the main file
add_executable(${CMAKE_PROJECT_NAME}_main main.c)
target_include_directories(${CMAKE_PROJECT_NAME}_main INTERFACE include)
target_link_libraries(${CMAKE_PROJECT_NAME}_main PRIVATE ${CMAKE_PROJECT_NAME})

# Download and unpack googletest at configure time
configure_file(CMakeLists.txt.in googletest-download/CMakeLists.txt)
execute_process(COMMAND ${CMAKE_COMMAND} -G "${CMAKE_GENERATOR}" .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "CMake step for googletest failed: ${result}")
endif()
execute_process(COMMAND ${CMAKE_COMMAND} --build .
  RESULT_VARIABLE result
  WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/googletest-download )
if(result)
  message(FATAL_ERROR "Build step for googletest failed: ${result}")
endif()

# Prevent overriding the parent project's compiler/linker
# settings on Windows
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)

# Add googletest directly to our build. This defines
# the gtest and gtest_main targets.
add_subdirectory(${CMAKE_CURRENT_BINARY_DIR}/googletest-src
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest-build
                 EXCLUDE_FROM_ALL)

# tests
add_executable(${CMAKE_PROJECT_NAME}_test test/testmain.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME}_test PUBLIC gtest_main ${CMAKE_PROJECT_NAME}_cifar10)
target_include_directories(${CMAKE_PROJECT_NAME}_test PUBLIC include)
