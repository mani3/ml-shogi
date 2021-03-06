cmake_minimum_required(VERSION 3.16)
project(ml_shogi)

if (${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
  set(CMAKE_CXX_FLAGS "-O2 -std=c++11 -Wall")
endif()

set(EXECUTABLE_OUTPUT_PATH "${CMAKE_SOURCE_DIR}/bin")

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Apery options
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -MMD -MP -msse4.2 -mbmi2 -mavx2 -DHAVE_SSE4 -DHAVE_SSE42 -DHAVE_BMI2 -DHAVE_AVX2")

find_package(TensorflowCC REQUIRED)
find_package(Protobuf REQUIRED)

include_directories(${CMAKE_SOURCE_DIR}/src)
aux_source_directory(${CMAKE_SOURCE_DIR}/src SOURCES)

# include_directories(${CMAKE_SOURCE_DIR}/libs/tensorflow_src)
include_directories(${CMAKE_SOURCE_DIR}/libs/flatbuffers/include)
include_directories(${CMAKE_SOURCE_DIR}/libs/libapery/include)
link_directories(${CMAKE_SOURCE_DIR}/libs/)

add_library(test_lib ${SOURCES})

target_link_libraries(test_lib TensorflowCC::TensorflowCC)
target_link_libraries(test_lib protobuf)

# target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libtensorflow-lite.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libflatbuffers.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libruy.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libfarmhash.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libfft2d_fftsg.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libfft2d_fftsg2d.a)
target_link_libraries(test_lib ${CMAKE_SOURCE_DIR}/libs/libapery/lib/libapery.a)

target_link_libraries(test_lib ${CMAKE_DL_LIBS})
target_link_libraries(test_lib pthread)

# add_executable(tf_lite_test ./test/tf_lite_test.cpp)
# target_link_libraries(tf_lite_test test_lib)

add_executable(model_test ./test/model_test.cpp)
target_link_libraries(model_test test_lib)

add_executable(example ./test/example.cpp)
target_link_libraries(example TensorflowCC::TensorflowCC)

add_executable(saved_model_test ./test/saved_model_test.cpp)
target_link_libraries(saved_model_test test_lib)

add_executable(concat_test ${CMAKE_SOURCE_DIR}/test/concat_test.cpp)
target_link_libraries(concat_test test_lib)


find_package(CUDA)
if(CUDA_FOUND)
  target_link_libraries(example ${CUDA_LIBRARIES})
endif()
