project(FRC-Kinect)

add_executable(${PROJECT_NAME} main.cpp)

#opencv4
set(OpenCV_DIR "$/home/trevor/vcpkg/installed/x64-linux/share/opencv4")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

#OpenGL
find_package(OpenGL REQUIRED)
include_directories(${OpenGL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} OpenGL::GL)

#freeglut
find_package(GLUT REQUIRED)
include_directories(${GLUT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} GLUT::GLUT)

#freenect
include_directories(/usr/include/libfreenect)
target_link_libraries(${PROJECT_NAME} freenect)

#include OpenCL
find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} OpenCL::OpenCL)