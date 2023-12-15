project(FRC-Kinect)

add_executable(${PROJECT_NAME} main.cpp)

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