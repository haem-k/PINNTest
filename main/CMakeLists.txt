add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/train.cpp)
target_link_libraries(Train PUBLIC PINNTest)