cmake_minimum_required(VERSION 3.16)
project(high_performance_face_recognition LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
find_package(wxWidgets COMPONENTS core base REQUIRED)
include(${wxWidgets_USE_FILE})

option(ENABLE_LIBTORCH "Link libtorch if available" OFF)

add_executable(hpfrec
  src/app.cpp
  src/face_database.cpp
  src/image_utils.cpp
  src/main_frame.cpp
)

target_include_directories(hpfrec PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(hpfrec PRIVATE ${OpenCV_LIBS} ${wxWidgets_LIBRARIES})

if(ENABLE_LIBTORCH)
  if(EXISTS "/opt/libtorch/lib/libtorch.so" OR EXISTS "/opt/libtorch/lib/libtorch_cpu.so")
    message(STATUS "Linking against libtorch in /opt/libtorch")
    target_include_directories(hpfrec PRIVATE /opt/libtorch/include)
    target_link_directories(hpfrec PRIVATE /opt/libtorch/lib)
    file(GLOB TORCH_LIBS "/opt/libtorch/lib/*.so")
    target_link_libraries(hpfrec PRIVATE ${TORCH_LIBS})
  else()
    message(WARNING "ENABLE_LIBTORCH set but /opt/libtorch not found. Skipping libtorch link.")
  endif()
endif()
