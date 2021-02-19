# Makefile for Lab5

CC = icc
CU_C = nvcc
OPTFLAGS = -Wall -O3
CUFLAGS = -lineinfo
PTXASFLAGS = -Xptxas -v
LDFLAGS = -L. -lhist-equ

H_FILES = hist-equ.h

LIB_SRC_C = histogram-equalization.c contrast-enhancement.c
LIB_SRC_CU = histogram.cu

OBJ_FILES = histogram-equalization.o contrast-enhancement.o

EXEC = main
SRC = main.c

all: main

main: $(OBJ_FILES)
	$(CC) $(OPTFLAGS) -o $(EXEC) $(SRC) $(LDFLAGS)

$(OBJ_FILES): $(LIB_SRC_C)
	$(CC) -c $(LIB_SRC_C)
	ar -cvq libhist-equ.a $(OBJ_FILES)

.PHONY: clean
clean:
	rm -f *.o *.a $(EXEC)
