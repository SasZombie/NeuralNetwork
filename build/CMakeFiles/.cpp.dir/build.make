# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.28

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/saszombie/Coding/NeuralNetwork

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/saszombie/Coding/NeuralNetwork/build

# Include any dependencies generated for this target.
include CMakeFiles/.cpp.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/.cpp.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/.cpp.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/.cpp.dir/flags.make

CMakeFiles/.cpp.dir/NN.cpp.o: CMakeFiles/.cpp.dir/flags.make
CMakeFiles/.cpp.dir/NN.cpp.o: /home/saszombie/Coding/NeuralNetwork/NN.cpp
CMakeFiles/.cpp.dir/NN.cpp.o: CMakeFiles/.cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/saszombie/Coding/NeuralNetwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/.cpp.dir/NN.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/.cpp.dir/NN.cpp.o -MF CMakeFiles/.cpp.dir/NN.cpp.o.d -o CMakeFiles/.cpp.dir/NN.cpp.o -c /home/saszombie/Coding/NeuralNetwork/NN.cpp

CMakeFiles/.cpp.dir/NN.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/.cpp.dir/NN.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saszombie/Coding/NeuralNetwork/NN.cpp > CMakeFiles/.cpp.dir/NN.cpp.i

CMakeFiles/.cpp.dir/NN.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/.cpp.dir/NN.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saszombie/Coding/NeuralNetwork/NN.cpp -o CMakeFiles/.cpp.dir/NN.cpp.s

CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o: CMakeFiles/.cpp.dir/flags.make
CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o: /home/saszombie/Coding/NeuralNetwork/NeuralNetwork.cpp
CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o: CMakeFiles/.cpp.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --progress-dir=/home/saszombie/Coding/NeuralNetwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o -MF CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o.d -o CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o -c /home/saszombie/Coding/NeuralNetwork/NeuralNetwork.cpp

CMakeFiles/.cpp.dir/NeuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Preprocessing CXX source to CMakeFiles/.cpp.dir/NeuralNetwork.cpp.i"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/saszombie/Coding/NeuralNetwork/NeuralNetwork.cpp > CMakeFiles/.cpp.dir/NeuralNetwork.cpp.i

CMakeFiles/.cpp.dir/NeuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green "Compiling CXX source to assembly CMakeFiles/.cpp.dir/NeuralNetwork.cpp.s"
	/usr/bin/g++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/saszombie/Coding/NeuralNetwork/NeuralNetwork.cpp -o CMakeFiles/.cpp.dir/NeuralNetwork.cpp.s

# Object files for target .cpp
_cpp_OBJECTS = \
"CMakeFiles/.cpp.dir/NN.cpp.o" \
"CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o"

# External object files for target .cpp
_cpp_EXTERNAL_OBJECTS =

.cpp: CMakeFiles/.cpp.dir/NN.cpp.o
.cpp: CMakeFiles/.cpp.dir/NeuralNetwork.cpp.o
.cpp: CMakeFiles/.cpp.dir/build.make
.cpp: CMakeFiles/.cpp.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color "--switch=$(COLOR)" --green --bold --progress-dir=/home/saszombie/Coding/NeuralNetwork/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable .cpp"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/.cpp.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/.cpp.dir/build: .cpp
.PHONY : CMakeFiles/.cpp.dir/build

CMakeFiles/.cpp.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/.cpp.dir/cmake_clean.cmake
.PHONY : CMakeFiles/.cpp.dir/clean

CMakeFiles/.cpp.dir/depend:
	cd /home/saszombie/Coding/NeuralNetwork/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/saszombie/Coding/NeuralNetwork /home/saszombie/Coding/NeuralNetwork /home/saszombie/Coding/NeuralNetwork/build /home/saszombie/Coding/NeuralNetwork/build /home/saszombie/Coding/NeuralNetwork/build/CMakeFiles/.cpp.dir/DependInfo.cmake "--color=$(COLOR)"
.PHONY : CMakeFiles/.cpp.dir/depend

