CC= gcc -g

SRCS=sparseval strealign headtrees
OBJS=sparseval.o sputil.o strealign.o headtrees.o
PROGS=sparseval.c sputil.c strealign.c headtrees.c

all: $(SRCS)

sparseval: sparseval.o sputil.o -lm 
	$(CC) -o sparseval sparseval.o sputil.o -lm

headtrees: headtrees.o sputil.o -lm 
	$(CC) -o headtrees headtrees.o sputil.o -lm

strealign: strealign.o sputil.o -lm 
	$(CC) -o strealign strealign.o sputil.o -lm

$(OBJS): $(INCS) Makefile
