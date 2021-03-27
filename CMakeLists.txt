# CMakeList.txt : CMake project for Perception_Miniprojekt, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

# Add source to this project's executable.
find_package(OpenCV REQUIRED)

add_executable (Perception_Miniprojekt "Perception_Miniprojekt.cpp" "Perception_Miniprojekt.h")

include_directories( ${OpenCV_INCLUDE_DIRS} )
target_link_libraries(Perception_Miniprojekt ${OpenCV_LIBS} )

# TODO: Add tests and install targets if needed.
