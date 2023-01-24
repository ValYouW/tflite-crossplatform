# For more information about using CMake with Android Studio, read the
# documentation: https://d.android.com/studio/projects/add-native-code.html

# Sets the minimum version of CMake required to build the native library.

cmake_minimum_required(VERSION 3.10.2)

# Declares and names the project.

project("tflite")

# opencv
set(OpenCV_STATIC ON)
set(OpenCV_DIR $ENV{OPENCV_ANDROID}/sdk/native/jni)
find_package (OpenCV REQUIRED)

# tensorflowlite
set(Tensorflowlite_DIR $ENV{TFLITE_DIST})
add_library(libtensorflowlite_c SHARED IMPORTED)
set_target_properties(libtensorflowlite_c PROPERTIES IMPORTED_LOCATION ${Tensorflowlite_DIR}/libs/android/${ANDROID_ABI}/libtensorflowlite_c.so)


# Creates and names a library, sets it as either STATIC
# or SHARED, and provides the relative paths to its source code.
# You can define multiple libraries, and CMake builds them for you.
# Gradle automatically packages shared libraries with your APK.

add_library( # Sets the name of the library.
    native-lib

    # Sets the library as a shared library.
    SHARED

    # Provides a relative path to your source file(s).
    ../../../../../ObjectDetection/ObjectDetector.cpp
    ../../../../../LinesDetector/LinesDetector.cpp
    ObjectDetectionActivity.cpp
    LinesDetectionActivity.cpp
)

target_include_directories(
    native-lib PRIVATE
    ../../../../../ObjectDetection
    ../../../../../LinesDetector
    ${Tensorflowlite_DIR}/include
)

# Searches for a specified prebuilt library and stores the path as a
# variable. Because CMake includes system libraries in the search path by
# default, you only need to specify the name of the public NDK library
# you want to add. CMake verifies that the library exists before
# completing its build.

find_library( # Sets the name of the path variable.
              log-lib

              # Specifies the name of the NDK library that
              # you want CMake to locate.
              log )

find_library(android-lib android)

# Specifies libraries CMake should link to your target library. You
# can link multiple libraries, such as libraries you define in this
# build script, prebuilt third-party libraries, or system libraries.

target_link_libraries( # Specifies the target library.
    native-lib

    libtensorflowlite_c
    ${OpenCV_LIBS}
    ${android-lib}

    # Links the target library to the log library
    # included in the NDK.
    ${log-lib} )