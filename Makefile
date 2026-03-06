CXX = g++
NVCC = nvcc

CXXFLAGS = -O3
NVFLAGS = -O3

TARGET = main

$(TARGET): main.cpp matrix.cu
	$(NVCC) $(NVFLAGS) $^ -o $(TARGET)

.PHONY: clean
clean:
	rm -f *.o $(TARGET)