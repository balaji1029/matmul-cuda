CXX = g++
NVCC = nvcc

CXXFLAGS = -O3
NVFLAGS = -O3
LDFLAGS = -lcublas

TARGET = main

$(TARGET): main.cpp matrix.cu
	$(NVCC) $(NVFLAGS) $^ -o $(TARGET) $(LDFLAGS)

.PHONY: clean
clean:
	rm -f *.o $(TARGET)