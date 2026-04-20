# Makefile
NVCC = nvcc
# Pascal architecture (MX130)
ARCH = -gencode arch=compute_50,code=sm_50
CFLAGS = -I./include
TARGET = glare_normalizer

all: $(TARGET)

$(TARGET): src/main.cu
	$(NVCC) $(ARCH) $(CFLAGS) src/main.cu -o $(TARGET)

clean:
	rm -f $(TARGET) data/processed/*.jpg
