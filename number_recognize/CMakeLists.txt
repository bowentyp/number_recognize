cmake_minimum_required(VERSION 3.5)
project(Demo_train)
find_package(OpenCV REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
set(CMAKE_CXX_STANDARD 14)
# link_directories(${OpenCV_LIBRARY_DIRS})
# target_link_libraries(Demo ${OpenCV_LIBS})

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
#message(STATUS "libraries path:${OpenCV_LIBRARY_DIRS}")
#message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")

include_directories(${OpenCV_INCLUDE_DIRS})
 

add_executable(Demo_train train_main.cpp)
target_link_libraries(Demo_train ${OpenCV_LIBS})
