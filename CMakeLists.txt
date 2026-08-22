cmake_minimum_required(VERSION 3.20)
project(debluring_cpp VERSION 1.0.0 LANGUAGES CXX)

option(DEBLUR_BUILD_TESTS "Build tests" ON)
option(DEBLUR_NATIVE_ARCH "Enable -march=native on GCC/Clang" OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(OpenMP)
find_library(FFTW3_LIBRARY NAMES fftw3 libfftw3.so.3 PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu REQUIRED)
find_library(FFTW3_THREADS_LIBRARY NAMES fftw3_threads libfftw3_threads.so.3 PATHS /lib/x86_64-linux-gnu /usr/lib/x86_64-linux-gnu)

add_library(deblur_lib
    src/deblur.cpp
    src/fft.cpp
    src/image_io.cpp
    src/image_ops.cpp
    src/optimization.cpp
    src/nonblind.cpp
)
target_include_directories(deblur_lib PUBLIC include PRIVATE src)
target_link_libraries(deblur_lib PUBLIC PNG::PNG JPEG::JPEG ${FFTW3_LIBRARY})
if(FFTW3_THREADS_LIBRARY)
    target_link_libraries(deblur_lib PUBLIC ${FFTW3_THREADS_LIBRARY})
endif()
if(OpenMP_CXX_FOUND)
    target_link_libraries(deblur_lib PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MSVC)
    target_compile_options(deblur_lib PRIVATE /O2 /W4)
else()
    target_compile_options(deblur_lib PRIVATE -O3 -Wall -Wextra -Wpedantic -ffp-contract=fast)
    if(DEBLUR_NATIVE_ARCH)
        target_compile_options(deblur_lib PRIVATE -march=native)
    endif()
endif()

include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_error)
if(ipo_supported AND CMAKE_BUILD_TYPE MATCHES "Release|RelWithDebInfo")
    set_property(TARGET deblur_lib PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

add_executable(deblur src/main.cpp)
target_link_libraries(deblur PRIVATE deblur_lib)

if(DEBLUR_BUILD_TESTS)
    enable_testing()
    add_executable(deblur_tests tests/test_deblur.cpp)
    target_link_libraries(deblur_tests PRIVATE deblur_lib)
    add_test(NAME deblur_unit_tests COMMAND deblur_tests)
endif()

install(TARGETS deblur RUNTIME DESTINATION bin)
