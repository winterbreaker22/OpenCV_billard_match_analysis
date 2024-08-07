# CMakeList.txt: progetto CMake. Includere l'origine e definire
# qui la logica specifica del progetto.

cmake_minimum_required (VERSION 3.8)

#project ("test")

# Aggiungere l'origine all'eseguibile di questo progetto.

project(test)

find_package(OpenCV REQUIRED)

add_executable(test main.cpp VideoProcessor.cpp BilliardAnalyzer.cpp)

#add_executable (test "main.cpp")

# Add tests and, if necessary, install destinations.
#find_package(OpenCV REQUIRED)

add_library(opencv INTERFACE IMPORTED)
set_target_properties(opencv PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${OpenCV_INCLUDE_DIRS}"
  INTERFACE_LINK_LIBRARIES "${OpenCV_LIBS}")

include_directories(${OpenCV_INCLUDE_DIRS})
#target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(test ${OpenCV_LIBS})
