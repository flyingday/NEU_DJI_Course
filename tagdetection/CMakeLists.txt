cmake_minimum_required(VERSION 2.6)
project(tagdetection)

find_package(OpenCV REQUIRED)

set(CMAKE_CXX_STANDARD 11)

add_executable(${PROJECT_NAME}
  src/main.cpp
  src/detection.cpp
)
target_include_directories(${PROJECT_NAME} PUBLIC
  ${OpenCV_DIR}
  ${CMAKE_CURRENT_LIST_DIR}/include
)
target_link_libraries(${PROJECT_NAME}
  ${OpenCV_LIBS}
)

install(TARGETS ${PROJECT_NAME} RUNTIME DESTINATION bin)
