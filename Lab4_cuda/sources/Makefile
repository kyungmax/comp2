NVCC=nvcc
NVCCFLAGS= -std=c++11
INCLUDE= -I./../libgputk
LIBgpuTK= -L./../build -L/usr/lib/x86_64-linux-gnu -lgputk -lcuda
LIBS= $(LIBgpuTK)

all: template_basic template_warp template_unified

template_basic:
	$(NVCC) $(NVCCFLAGS) template_basic.cu $(INCLUDE) $(LIBS) -o Reduction_Template

template_unified:
	$(NVCC) $(NVCCFLAGS) template_unified.cu $(INCLUDE) $(LIBS) -o Reduction_Template

template_warp:
	$(NVCC) $(NVCCFLAGS) template_warp.cu $(INCLUDE) $(LIBS) -o Reduction_Template

solution_basic:
	$(NVCC) $(NVCCFLAGS) solution_basic.cu $(INCLUDE) $(LIBS) -o Reduction_Solution

solution_unified:
	$(NVCC) $(NVCCFLAGS) solution_unified.cu $(INCLUDE) $(LIBS) -o Reduction_Solution

solution_warp:
	$(NVCC) $(NVCCFLAGS) solution_warp.cu $(INCLUDE) $(LIBS) -o Reduction_Solution

solutions: solution_basic solution_warp solution_unified

clean:
	rm -f Reduction_Template Reduction_Solution
