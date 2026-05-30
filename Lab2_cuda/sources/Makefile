NVCC=nvcc
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -L/usr/lib/x86_64-linux-gnu -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: template

template:
	$(NVCC) -std=c++11 template.cu $(INCLUDE) $(LIBS) -o BatchedMatMul_template

solution:
	$(NVCC) -std=c++11 solution.cu $(INCLUDE) $(LIBS) -o BatchedMatMul_solution

dataset_generator:
	$(NVCC) -std=c++11 dataset_generator.cpp $(INCLUDE) $(LIBS) -o dataset_generator

dist:
	bash package_student.sh

clean:
	rm -f BatchedMatMul_template BatchedMatMul_solution dataset_generator
