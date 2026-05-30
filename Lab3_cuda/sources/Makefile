NVCC=nvcc
NVCCFLAGS= -std=c++11
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -L/usr/lib/x86_64-linux-gnu -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: convolution

convolution:
	$(NVCC) $(NVCCFLAGS) template_convolution.cu $(INCLUDE) $(LIBS) -o Convolution_template

solution:
	$(NVCC) $(NVCCFLAGS) solution_convolution.cu $(INCLUDE) $(LIBS) -o Convolution_solution

dataset_generator_convolution:
	$(NVCC) $(NVCCFLAGS) dataset_generator_convolution.cpp $(INCLUDE) $(LIBS) -o dataset_generator_convolution

generators: dataset_generator_convolution

clean:
	rm -f Convolution_template Convolution_solution dataset_generator_convolution
