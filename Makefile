CXX = g++
NVCC = nvcc

CXXFLAGS = -O3
NVFLAGS = -O3

TARGET = main

$(TARGET): main.o matrix.o
	$(NVCC) $(NVFLAGS) main.o matrix.o -o $(TARGET)

matrix.o: matrix.cu
	$(NVCC) $(NVFLAGS) -c matrix.cu -o matrix.o

main.o: main.cpp
	$(CXX) $(CXXFLAGS) -c main.cpp -o main.o

.PHONY: clean
clean:
	rm -f *.o $(TARGET)