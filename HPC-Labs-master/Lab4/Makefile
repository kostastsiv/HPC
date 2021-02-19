# Makefile for .cu files

CC = gcc
NVCC = nvcc
# Edw vale kai -Wall an niw8eis brave, 8a bgalei xaza warnings
CCFLAGS = -Xcompiler -O4
# Shows some info regarding register & memory usage, probably not useful
PTXFLAGS = -Xptxas -v
# Also probably not useful
OPTFLAGS = -lineinfo
EXECUTABLES = Convolution2D_OPT Convolution2D_OPT_Blocked Convolution2D_OPT_Streams

all: $(EXECUTABLES)

%: %.cu
	$(NVCC) $(PTXFLAGS) --resource-usage $(CCFLAGS) $< -o $@

.PHONY: clean

clean:
	rm -f $(EXECUTABLES)
