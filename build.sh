g++ $1.cpp -o src/$1 NeuralNetwork.cpp -O3 -std=c++23 -lraylib -lGL -lm -lpthread -ldl -lm -lrt -lX11 -Wall -Wextra -Wformat-nonliteral -Wcast-align -Wpointer-arith -Wmissing-declarations -Winline -Wundef -Wcast-qual -Wshadow -Wwrite-strings -Wno-unused-parameter -Wfloat-equal -pedantic #-fsanitize=address -fsanitize=leak
