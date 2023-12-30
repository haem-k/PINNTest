# Build GLM
cd /home/haemin/Documents/aOpenGL/ext/glm
mkdir build
cd build
cmake ..
make -j

# Build GLFW
cd /home/haemin/Documents/aOpenGL/ext/glfw
mkdir build
cd build
cmake ..
make -j

# Build FREETYPE
cd /home/haemin/Documents/aOpenGL/ext/freetype
mkdir build
cd build
cmake -D BUILD_SHARED_LIBS=true -D CMAKE_BUILD_TYPE=Release ..
make -j

# Build aOpenGL
cd /home/haemin/Documents/aOpenGL/build
cmake ..
make -j

# Build aLibTorch
cd /home/haemin/Documents/aLibTorch/build
cmake ..
make -j


