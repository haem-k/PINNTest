add_executable(Train ${CMAKE_CURRENT_SOURCE_DIR}/train.cpp)
target_link_libraries(Train PUBLIC PINNTest)

add_executable(Test ${CMAKE_CURRENT_SOURCE_DIR}/test_vis.cpp)
target_link_libraries(Test PUBLIC PINNTest)