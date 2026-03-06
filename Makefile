
TARGET: main

main: main.o matrix.o
	nvcc main.o matrix.o -o main

matrix.o: matrix.cu
	nvcc -c matrix.cu -o matrix.o

main.o: main.cpp
	g++ -c main.cpp -o main.o
