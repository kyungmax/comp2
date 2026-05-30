NVCC=nvcc
NVCCFLAGS= -std=c++11
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -L/usr/lib/x86_64-linux-gnu -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: vecadd matmul

vecadd:
	$(NVCC) $(NVCCFLAGS) template_vecadd.cu $(INCLUDE) $(LIBS) -o VectorAdd_template

matmul:
	$(NVCC) $(NVCCFLAGS) template_matmul.cu $(INCLUDE) $(LIBS) -o TiledMatMul_template

dataset_generator_vecadd:
	$(NVCC) $(NVCCFLAGS) dataset_generator_vecadd.cpp $(INCLUDE) $(LIBS) -o dataset_generator_vecadd

dataset_generator_matmul:
	$(NVCC) $(NVCCFLAGS) dataset_generator_matmul.cpp $(INCLUDE) $(LIBS) -o dataset_generator_matmul

generators: dataset_generator_vecadd dataset_generator_matmul

clean:
	rm -f VectorAdd_template TiledMatMul_template dataset_generator_vecadd dataset_generator_matmul
