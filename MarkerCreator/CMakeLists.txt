project(FRC-AprilTagMaker)

add_executable(${PROJECT_NAME} main.cpp)

#opencv4
set(OpenCV_DIR "$/home/trevor/vcpkg/installed/x64-linux/share/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})