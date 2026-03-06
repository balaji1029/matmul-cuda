CXX = nvcc
CXXFLAGS = -O3

TARGET: main

main: main.o matrix.o
	$(CXX) $(CXXFLAGS) main.o matrix.o -o main

matrix.o: matrix.cu
	$(CXX) $(CXXFLAGS) -c matrix.cu -o matrix.o

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o