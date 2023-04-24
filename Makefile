CC          := nvcc
CFLAGS      := -O3 -c -arch=sm_60
LFLAGS      := -O3 -arch=sm_60
ALL         := brandes-cuda.exe

all : $(ALL)

skip_1deg : CFLAGS += -DSKIP_1DEG
skip_1deg : test

skip_bfs : CFLAGS += -DSKIP_BFS 
skip_bfs : test

skip_stride : CFLAGS += -DSKIP_STRIDE 
skip_stride : test

test : CFLAGS += -DDETAILED_METRICS -DPRINT_PROGRESS 
test : $(ALL)

%.exe : %.o
	$(CC) $(LFLAGS) -o brandes $<


%.o : %.cu
	$(CC) $(CFLAGS) $<

clean :
	rm -f *.o *.out *.err brandes

