add_library(PINNTest SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/src/network.cpp
)

target_include_directories(PINNTest PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(PINNTest PUBLIC aLibTorch aOpenGL)
